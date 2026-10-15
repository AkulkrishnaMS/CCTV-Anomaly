"""YOLOv8/v11 + ByteTrack tracking wrapper, with false-positive reduction:
per-class confidence thresholds, minimum box size filter, person aspect-
ratio check, and class-history smoothing."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter

import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# ITEM 1: Per-class confidence thresholds (module-level, above the classes)
# ============================================================
CLASS_CONF_THRESHOLDS = {
    "person": 0.5,
    "knife": 0.65,
    "scissors": 0.65,
    "backpack": 0.55,
    "handbag": 0.55,
    "suitcase": 0.55,
    "dog": 0.55,
    "cat": 0.55,
    "car": 0.5,
    "motorcycle": 0.5,
    "bicycle": 0.5,
}

def passes_class_threshold(class_name: str, conf: float) -> bool:
    return conf >= CLASS_CONF_THRESHOLDS.get(class_name, 0.55)


# ============================================================
# ITEM 2: Minimum bounding-box size filter (module-level)
# ============================================================
MIN_BOX_AREA_FRACTION = 0.0015  # ~0.15% of frame area minimum

def box_area_fraction(x1, y1, x2, y2, frame_width, frame_height):
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    return box_area / frame_area


# ============================================================
# ITEM 3: Aspect ratio sanity check for "person" (module-level)
# ============================================================
def is_plausible_person_box(x1, y1, x2, y2, min_ratio=1.2, max_ratio=5.0):
    w, h = x2 - x1, y2 - y1
    if w <= 0:
        return False
    ratio = h / w
    return min_ratio <= ratio <= max_ratio


@dataclass
class TrackState:
    track_id: int
    class_name: str
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    first_seen_frame: int = 0
    last_seen_frame: int = 0

    # ============================================================
    # ITEM 4: Class-history smoothing (field on TrackState)
    # ============================================================
    class_history: deque = field(default_factory=lambda: deque(maxlen=15))

    def update(self, new_centroid: Tuple[float, float], frame_idx: int,
               new_bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        prev_cx, prev_cy = self.centroid
        new_cx, new_cy = new_centroid
        self.velocity = (new_cx - prev_cx, new_cy - prev_cy)
        self.centroid = new_centroid
        if new_bbox is not None:
            self.bbox = new_bbox
        self.history.append(new_centroid)
        self.last_seen_frame = frame_idx

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def track_length_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame

    def is_trusted(self, min_frames: int = 5) -> bool:
        return self.track_length_frames >= min_frames

    # ITEM 4 (continued): majority-vote class smoothing method
    def stable_class_name(self) -> str:
        if not self.class_history:
            return self.class_name
        return Counter(self.class_history).most_common(1)[0][0]


class ObjectTracker:
    def __init__(self, model: YOLO, class_ids: List[int], device: torch.device,
                 conf: float = 0.55, iou: float = 0.5):
        self.model = model
        self.class_ids = class_ids
        self.device = device
        self.conf = conf
        self.iou = iou
        self.tracks: Dict[int, TrackState] = {}

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[TrackState]:
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=self.class_ids, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )
        active_tracks: List[TrackState] = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return active_tracks

        frame_h, frame_w = frame.shape[:2]

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        # ============================================================
        # ITEM 5: Combined order -- confidence -> size -> aspect ratio
        # -> ByteTrack ID assignment (already done above) -> smoothing
        # ============================================================
        for box, tid, cls_id, conf in zip(boxes_xyxy, track_ids, class_ids, confs):
            x1, y1, x2, y2 = box
            class_name = self.model.names.get(int(cls_id), str(cls_id))

            # --- ITEM 1 applied here ---
            if not passes_class_threshold(class_name, conf):
                continue

            # --- ITEM 2 applied here ---
            if box_area_fraction(x1, y1, x2, y2, frame_w, frame_h) < MIN_BOX_AREA_FRACTION:
                continue

            # --- ITEM 3 applied here ---
            if class_name == "person" and not is_plausible_person_box(x1, y1, x2, y2):
                continue

            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(
                    track_id=int(tid), class_name=class_name,
                    centroid=centroid, bbox=(x1, y1, x2, y2),
                    first_seen_frame=frame_idx, last_seen_frame=frame_idx,
                )
                self.tracks[tid].history.append(centroid)
            else:
                self.tracks[tid].update(centroid, frame_idx, new_bbox=(x1, y1, x2, y2))

            # --- ITEM 4 applied here ---
            self.tracks[tid].class_history.append(class_name)
            self.tracks[tid].class_name = self.tracks[tid].stable_class_name()

            active_tracks.append(self.tracks[tid])

        return active_tracks

    def get_trusted_tracks(self, tracks: List[TrackState], min_frames: int = 5) -> List[TrackState]:
        return [t for t in tracks if t.is_trusted(min_frames)]
