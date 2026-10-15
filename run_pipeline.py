#!/usr/bin/env python3
"""
run_pipeline.py — Context-Aware Smart Home Surveillance, Final Pipeline

Usage:
    python run_pipeline.py --video /app/datasets/input_videos/sample.mp4 \
        --occupancy vacant --time 23:30 --email-alerts

Runs: Video -> MNAD (anomaly score) -> YOLO/ByteTrack -> Context Engine ->
Risk Engine -> Explainability Engine -> annotated video + CSV + JSON + plot.
Optionally sends an email alert when risk crosses a threshold.
"""

import os
import sys
import time
import json
import argparse
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime

import cv2
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SmartHomeSurveillance.Pipeline")


# ============================================================
# Paths / Config
# ============================================================
@dataclass
class ProjectPaths:
    root: Path
    app: Path = field(init=False)
    config: Path = field(init=False)
    datasets: Path = field(init=False)
    mnad_repo: Path = field(init=False)
    checkpoints: Path = field(init=False)
    outputs: Path = field(init=False)
    logs: Path = field(init=False)

    def __post_init__(self) -> None:
        self.app = self.root / "app"
        self.config = self.root / "config"
        self.datasets = self.root / "datasets"
        self.mnad_repo = self.root / "MNAD"
        self.checkpoints = self.root / "checkpoints"
        self.outputs = self.root / "outputs"
        self.logs = self.root / "logs"


def load_config(project_root: Path) -> dict:
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================
# CLI arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Smart Home Surveillance final pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--project-root", type=str, default="/app", help="Project root directory")
    parser.add_argument("--occupancy", type=str, default="vacant", choices=["vacant", "occupied"])
    parser.add_argument("--time", type=str, default="23:30", help="Simulated time, HH:MM (24h)")
    parser.add_argument("--entry-point", type=str, default=None,
                         help="Optional entry point as 'name,x,y' e.g. 'front_door,450,300'")
    parser.add_argument("--conf", type=float, default=0.55, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    parser.add_argument("--multiple-people-thresh", type=int, default=2)
    parser.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    parser.add_argument("--high-risk-thresh", type=float, default=70.0)

    # --- Email alert arguments (default to environment variables, so
    # credentials never need to be typed on the command line or committed
    # to source control -- see .env file / --env-file usage) ---
    parser.add_argument("--email-alerts", action="store_true", help="Enable email alerts on high-risk events")
    parser.add_argument("--smtp-server", type=str, default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--sender-email", type=str, default=os.environ.get("ALERT_SENDER_EMAIL"))
    parser.add_argument("--sender-password", type=str, default=os.environ.get("ALERT_SENDER_APP_PASSWORD"))
    parser.add_argument("--owner-email", type=str, default=os.environ.get("ALERT_OWNER_EMAIL"))
    parser.add_argument("--email-cooldown-sec", type=float, default=120.0,
                         help="Minimum seconds between consecutive email alerts")

    return parser.parse_args()


# ============================================================
# Email alert helper
# ============================================================
def send_alert_email(args, risk_score, timestamp_str, reasons, context_labels_str, video_name):
    if not args.email_alerts:
        return
    if not (args.sender_email and args.sender_password and args.owner_email):
        logger.warning("Email alerts enabled but sender/owner credentials not fully provided -- skipping.")
        return

    msg = MIMEMultipart()
    msg["From"] = args.sender_email
    msg["To"] = args.owner_email
    msg["Subject"] = f"[Smart Home Alert] High risk detected ({risk_score:.0f}/100) - {video_name}"

    body = (
        f"A high-risk event was detected in your smart home surveillance feed.\n\n"
        f"Video: {video_name}\n"
        f"Time: {timestamp_str}\n"
        f"Risk Score: {risk_score:.0f}/100\n\n"
        f"Context: {context_labels_str}\n\n"
        f"Reasons:\n" + "\n".join(f"  - {r}" for r in reasons) + "\n\n"
        f"This is an automated alert from your Smart Home Surveillance system."
    )
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(args.smtp_server, args.smtp_port) as server:
            server.starttls()
            server.login(args.sender_email, args.sender_password)
            server.send_message(msg)
        logger.info(f"Alert email sent to {args.owner_email} (risk={risk_score:.0f})")
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    PATHS = ProjectPaths(root=Path(args.project_root))
    sys.path.insert(0, str(PATHS.mnad_repo))
    sys.path.insert(0, str(PATHS.app.parent))

    from app.mnad_detector import MNADDetector
    from app.object_tracker import ObjectTracker, TrackState
    from app.context_risk_explain import (
        ContextEngine, RiskEngine, RiskWeights, ExplainabilityEngine, ContextLabel,
    )
    from ultralytics import YOLO

    CONFIG = load_config(PATHS.root)
    DEVICE = torch.device(CONFIG.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")

    INPUT_VIDEO_PATH = Path(args.video)
    if not INPUT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {INPUT_VIDEO_PATH}")

    hh, mm = map(int, args.time.split(":"))
    CURRENT_TIME = dtime(hh, mm)
    OCCUPANCY_STATUS = args.occupancy

    # Base datetime used to compute a real, advancing clock time per frame
    base_datetime = datetime.combine(datetime.today(), CURRENT_TIME)
    last_email_sent_time = [0.0]

    logger.info(f"Simulated occupancy: {OCCUPANCY_STATUS}")
    logger.info(f"Base timestamp for this run: {base_datetime}")

    ENTRY_POINTS = {}
    if args.entry_point:
        name, x, y = args.entry_point.split(",")
        ENTRY_POINTS[name] = (float(x), float(y))
        logger.info(f"Entry point configured: {ENTRY_POINTS}")

    # ---- Video info ----
    cap_check = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    assert cap_check.isOpened(), f"OpenCV could not open {INPUT_VIDEO_PATH}"
    VIDEO_FPS = cap_check.get(cv2.CAP_PROP_FPS) or 25.0
    VIDEO_WIDTH = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDEO_HEIGHT = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VIDEO_FRAME_COUNT = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_check.release()
    logger.info(f"Video: {INPUT_VIDEO_PATH.name} | {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS:.2f}fps | {VIDEO_FRAME_COUNT} frames")

    # ---- MNAD (optional -- falls back to 0.0 if no checkpoint) ----
    mnad_model_path = PATHS.checkpoints / "ped2" / "model.pth"
    mnad_items_path = PATHS.checkpoints / "ped2" / "m_items.pt"
    video_mnad_scores = None
    MNAD_TIME_STEP = 4

    video_frames_dir = PATHS.datasets / "input_videos" / f"{INPUT_VIDEO_PATH.stem}_frames" / "clip_001"
    if mnad_model_path.exists() and mnad_items_path.exists():
        logger.info("Loading MNAD detector and extracting frames for scoring...")
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = list(video_frames_dir.glob("*.jpg"))
        if len(existing_frames) != VIDEO_FRAME_COUNT:
            cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
            width_pad = len(str(VIDEO_FRAME_COUNT - 1))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(video_frames_dir / f"{idx:0{width_pad}d}.jpg"), frame)
                idx += 1
            cap.release()
            logger.info(f"Extracted {idx} frames for MNAD scoring")

        mnad_detector = MNADDetector(
            model_path=mnad_model_path, m_items_path=mnad_items_path,
            method="pred", device=DEVICE,
        )
        scores_by_clip = mnad_detector.score_frames_folder(video_frames_dir.parent)
        raw_scores = scores_by_clip["clip_001"]

        video_mnad_scores = np.full(
            VIDEO_FRAME_COUNT, raw_scores[0] if len(raw_scores) else 0.0, dtype=float
        )
        video_mnad_scores[MNAD_TIME_STEP:MNAD_TIME_STEP + len(raw_scores)] = raw_scores
        logger.info(
            f"MNAD scoring complete: {len(raw_scores)} scored frames, "
            f"range [{video_mnad_scores.min():.3f}, {video_mnad_scores.max():.3f}]"
        )
    else:
        logger.warning("No MNAD checkpoint found -- using mnad_score=0.0 for every frame.")

    # ---- YOLO + ByteTrack ----
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(DEVICE)

    TARGET_CLASS_NAMES = [
        "person", "knife", "scissors", "backpack", "handbag", "suitcase",
        "dog", "cat", "car", "motorcycle", "bicycle",
    ]
    NAME_TO_ID = {name: idx for idx, name in yolo_model.names.items()}
    TARGET_CLASS_IDS = [NAME_TO_ID[n] for n in TARGET_CLASS_NAMES if n in NAME_TO_ID]
    logger.info(f"YOLO model: {args.yolo_model} | classes: {TARGET_CLASS_NAMES}")

    tracker = ObjectTracker(model=yolo_model, class_ids=TARGET_CLASS_IDS, device=DEVICE,
                             conf=args.conf, iou=args.iou)

    # ---- Pass 1: tracking ----
    logger.info("Pass 1/2: object detection + tracking...")
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    log_rows = []
    frame_idx = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracks = tracker.process_frame(frame, frame_idx)
        tracks = tracker.get_trusted_tracks(tracks, min_frames=5)
        for t in tracks:
            log_rows.append({
                "frame_index": frame_idx, "track_id": t.track_id, "class_name": t.class_name,
                "centroid_x": t.centroid[0], "centroid_y": t.centroid[1],
                "velocity_x": t.velocity[0], "velocity_y": t.velocity[1], "speed": t.speed,
            })
        if frame_idx % 100 == 0:
            logger.info(f"  [Pass 1] frame {frame_idx}/{VIDEO_FRAME_COUNT}")
        frame_idx += 1
    cap.release()
    tracking_log = pd.DataFrame(log_rows)
    logger.info(f"Pass 1 complete in {time.time()-start_time:.1f}s: {len(tracking_log)} detections")

    # ---- Context + Risk engines ----
    context_engine = ContextEngine(tracking_log, multiple_people_thresh=args.multiple_people_thresh,
                                    entry_points=ENTRY_POINTS)
    risk_engine = RiskEngine(weights=RiskWeights())
    explain_engine = ExplainabilityEngine()

    # ---- Drawing helpers ----
    TRACK_COLOR_PALETTE = [
        (66, 135, 245), (245, 66, 93), (66, 245, 138), (245, 197, 66),
        (197, 66, 245), (66, 245, 224), (245, 138, 66), (138, 66, 245),
    ]

    def color_for_track(track_id):
        return TRACK_COLOR_PALETTE[track_id % len(TRACK_COLOR_PALETTE)]

    def risk_color(risk_score):
        if risk_score < 40:
            return (0, 200, 0)
        elif risk_score < 70:
            return (0, 200, 255)
        return (0, 0, 255)

    def draw_full_annotation(frame, tracks, risk_score, mnad_score, context_labels_str,
                              top_reason, frame_idx, current_datetime):
        out = frame.copy()
        for t in tracks:
            color = color_for_track(t.track_id)
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cx, cy = int(t.centroid[0]), int(t.centroid[1])
            pts = list(t.history)
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, p1, p2, color, 1, lineType=cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 4, color, -1)
            vx, vy = t.velocity
            cv2.arrowedLine(out, (cx, cy), (int(cx + vx*3), int(cy + vy*3)), color, 2, tipLength=0.3)
            cv2.putText(out, f"ID {t.track_id} {t.class_name}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        banner_color = risk_color(risk_score)
        cv2.rectangle(out, (0, 0), (out.shape[1], 65), (30, 30, 30), -1)
        time_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, f"{time_str}  |  Risk: {risk_score:.0f}  |  MNAD: {mnad_score:.2f}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"Context: {context_labels_str}", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)

        cv2.rectangle(out, (0, out.shape[0]-30), (out.shape[1], out.shape[0]), (30, 30, 30), -1)
        cv2.putText(out, top_reason[:100], (10, out.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ---- Pass 2: context + risk + draw + email alerts ----
    logger.info("Pass 2/2: context, risk, explanation, annotation...")
    tracker_for_draw = ObjectTracker(model=yolo_model, class_ids=TARGET_CLASS_IDS, device=DEVICE,
                                      conf=args.conf, iou=args.iou)

    PATHS.outputs.mkdir(parents=True, exist_ok=True)
    annotated_video_path = PATHS.outputs / f"{INPUT_VIDEO_PATH.stem}_annotated.mp4"
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    frame_idx = 0
    events = []
    risk_rows = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_seconds = frame_idx / VIDEO_FPS
        current_frame_datetime = base_datetime + timedelta(seconds=elapsed_seconds)

        tracks = tracker_for_draw.process_frame(frame, frame_idx)
        mnad_score = float(video_mnad_scores[frame_idx]) if video_mnad_scores is not None else 0.0

        ctx = context_engine.analyze_frame(
            frame_idx, occupancy=OCCUPANCY_STATUS, current_time=CURRENT_TIME,
            frame_width=VIDEO_WIDTH, frame_height=VIDEO_HEIGHT,
        )
        risk = risk_engine.compute(mnad_score, ctx)
        reasons = explain_engine.explain(risk, ctx, mnad_score)
        top_reason = reasons[0] if reasons else "No significant risk factors detected"
        context_labels_str = ", ".join(l.value for l in ctx.labels) if ctx.labels else "None"

        annotated = draw_full_annotation(frame, tracks, risk, mnad_score, context_labels_str,
                                          top_reason, frame_idx, current_frame_datetime)
        writer.write(annotated)

        class_counts = {}
        for t in tracks:
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1
        class_breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))

        risk_rows.append({
            "frame_index": frame_idx, "timestamp_sec": round(frame_idx / VIDEO_FPS, 2),
            "clock_time": current_frame_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "mnad_score": round(mnad_score, 4), "person_count": ctx.person_count,
            "object_count": ctx.object_count, "yolo_class_breakdown": class_breakdown,
            "context_labels": context_labels_str, "occupancy": ctx.occupancy,
            "is_night": ctx.is_night, "risk_score": round(risk, 2),
            "reasons": " | ".join(reasons),
        })

        if risk >= args.high_risk_thresh:
            events.append({
                "frame_index": frame_idx, "timestamp_sec": round(frame_idx / VIDEO_FPS, 2),
                "clock_time": current_frame_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "mnad_score": round(mnad_score, 4), "risk_score": risk,
                "context_labels": [l.value for l in ctx.labels], "reasons": reasons,
            })

            now = time.time()
            if now - last_email_sent_time[0] >= args.email_cooldown_sec:
                send_alert_email(
                    args, risk, current_frame_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    reasons, context_labels_str, INPUT_VIDEO_PATH.name,
                )
                last_email_sent_time[0] = now

        if frame_idx % 100 == 0:
            logger.info(f"  [Pass 2] frame {frame_idx}/{VIDEO_FRAME_COUNT}")
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info(f"Pass 2 complete in {time.time()-start_time:.1f}s")
    logger.info(f"Annotated video saved -> {annotated_video_path}")

    risk_df = pd.DataFrame(risk_rows)

    # ---- Save outputs ----
    risk_csv_path = PATHS.outputs / f"{INPUT_VIDEO_PATH.stem}_risk_scores.csv"
    risk_df.to_csv(risk_csv_path, index=False)

    events_json_path = PATHS.outputs / f"{INPUT_VIDEO_PATH.stem}_events.json"
    with open(events_json_path, "w") as f:
        json.dump(events, f, indent=2)

    anomaly_plot_path = PATHS.outputs / f"{INPUT_VIDEO_PATH.stem}_risk_plot.png"
    plt.figure(figsize=(12, 4))
    plt.plot(risk_df["frame_index"], risk_df["risk_score"], linewidth=1)
    plt.axhline(args.high_risk_thresh, color="red", linestyle="--", linewidth=0.8, label="High-risk threshold")
    plt.xlabel("Frame Index"); plt.ylabel("Risk Score")
    plt.title(f"Risk Score Over Time — {INPUT_VIDEO_PATH.name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(anomaly_plot_path, dpi=120)
    plt.close()

    logger.info("=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Annotated video:  {annotated_video_path}")
    logger.info(f"Risk scores CSV:  {risk_csv_path}")
    logger.info(f"Events JSON:      {events_json_path} ({len(events)} high-risk events)")
    logger.info(f"Risk plot:        {anomaly_plot_path}")
    logger.info(f"Mean risk score:  {risk_df['risk_score'].mean():.1f}")
    logger.info(f"Max risk score:   {risk_df['risk_score'].max():.1f}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
