"""Context, Risk, and Explainability engines.

Includes: base context labels, entry-point approach detection, possible-
object-taken detection, non-linear MNAD score fusion (muted at low/moderate
scores to counter domain-mismatch noise, boosted at high scores so genuine
anomalies like fighting still raise risk), and possible-altercation
detection (independent of MNAD, since MNAD trained on Ped2 does not
reliably flag fighting)."""

from dataclasses import dataclass, field
from datetime import time as dtime
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def is_night(t: dtime, night_start: dtime = dtime(21, 0), night_end: dtime = dtime(6, 0)) -> bool:
    if night_start <= night_end:
        return night_start <= t <= night_end
    return t >= night_start or t <= night_end


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_angle = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class ContextLabel(str, Enum):
    PERSON_PRESENT = "Person present"
    MULTIPLE_PEOPLE = "Multiple people"
    LOITERING = "Loitering"
    UNKNOWN_OBJECT = "Unknown object"
    ABANDONED_OBJECT = "Abandoned object"
    NIGHT_ACTIVITY = "Night activity"
    NO_ACTIVITY = "No activity"
    ZONE_ENTRY = "Zone entry (zones not configured)"
    APPROACHING_ENTRY = "Approaching entry point"
    POSSIBLE_OBJECT_TAKEN = "Possible object taken by different person"
    POSSIBLE_ALTERCATION = "Possible physical altercation (close contact, high relative motion)"


EXPECTED_CLASSES = {
    "person", "knife", "scissors", "backpack", "handbag", "suitcase",
    "dog", "cat", "car", "motorcycle", "bicycle",
}
OBJECT_CLASSES = EXPECTED_CLASSES - {"person"}


@dataclass
class FrameContext:
    frame_index: int
    labels: List[ContextLabel] = field(default_factory=list)
    person_count: int = 0
    object_count: int = 0
    loitering_track_ids: List[int] = field(default_factory=list)
    unknown_class_names: List[str] = field(default_factory=list)
    abandoned_track_ids: List[int] = field(default_factory=list)
    approaching_entry: List[Tuple[int, str]] = field(default_factory=list)
    possibly_taken_objects: List[Tuple[int, dict]] = field(default_factory=list)
    altercation_pairs: List[Tuple[int, int]] = field(default_factory=list)
    is_night: bool = False
    occupancy: str = "unknown"


class ContextEngine:
    def __init__(
        self, tracking_log: pd.DataFrame,
        loiter_speed_thresh: float = 2.0,
        loiter_duration_frames: int = 90,
        abandoned_object_frames: int = 150,
        abandoned_proximity_px: float = 150.0,
        multiple_people_thresh: int = 2,
        entry_points: Dict[str, Tuple[float, float]] = None,
        approach_angle_threshold_deg: float = 30.0,
        approach_min_speed: float = 1.5,
        altercation_proximity_px: float = 80.0,
        altercation_relative_speed_thresh: float = 15.0,
        altercation_sustained_frames: int = 10,
    ):
        self.df = tracking_log
        self.loiter_speed_thresh = loiter_speed_thresh
        self.loiter_duration_frames = loiter_duration_frames
        self.abandoned_object_frames = abandoned_object_frames
        self.abandoned_proximity_px = abandoned_proximity_px
        self.multiple_people_thresh = multiple_people_thresh
        self.entry_points = entry_points or {}
        self.approach_angle_threshold_deg = approach_angle_threshold_deg
        self.approach_min_speed = approach_min_speed
        self.altercation_proximity_px = altercation_proximity_px
        self.altercation_relative_speed_thresh = altercation_relative_speed_thresh
        self.altercation_sustained_frames = altercation_sustained_frames

        self._track_spans = (
            self.df.groupby("track_id")
            .agg(class_name=("class_name", "first"),
                 first_frame=("frame_index", "min"),
                 last_frame=("frame_index", "max"),
                 avg_speed=("speed", "mean"))
            .reset_index()
        )

        self._possibly_taken = self._possibly_taken_objects()

    def _loitering_tracks_at(self, frame_idx: int) -> List[int]:
        active = self._track_spans[
            (self._track_spans["class_name"] == "person")
            & (self._track_spans["first_frame"] <= frame_idx)
            & (self._track_spans["last_frame"] >= frame_idx)
        ]
        duration = frame_idx - active["first_frame"]
        is_loitering = (duration >= self.loiter_duration_frames) & (active["avg_speed"] <= self.loiter_speed_thresh)
        return active[is_loitering]["track_id"].tolist()

    def _abandoned_objects_at(self, frame_idx: int, frame_rows: pd.DataFrame) -> List[int]:
        obj_rows = frame_rows[frame_rows["class_name"].isin(OBJECT_CLASSES)]
        person_rows = frame_rows[frame_rows["class_name"] == "person"]
        abandoned = []
        for _, obj in obj_rows.iterrows():
            span = self._track_spans[self._track_spans["track_id"] == obj["track_id"]].iloc[0]
            duration_present = frame_idx - span["first_frame"]
            if duration_present < self.abandoned_object_frames:
                continue
            if person_rows.empty:
                abandoned.append(int(obj["track_id"]))
                continue
            dists = np.hypot(person_rows["centroid_x"] - obj["centroid_x"],
                              person_rows["centroid_y"] - obj["centroid_y"])
            if dists.min() > self.abandoned_proximity_px:
                abandoned.append(int(obj["track_id"]))
        return abandoned

    def _approaching_entry_tracks_at(self, frame_rows: pd.DataFrame) -> List[Tuple[int, str]]:
        if not self.entry_points:
            return []
        person_rows = frame_rows[frame_rows["class_name"] == "person"]
        results = []
        for _, row in person_rows.iterrows():
            speed = row["speed"]
            if speed < self.approach_min_speed:
                continue
            velocity = (row["velocity_x"], row["velocity_y"])
            centroid = (row["centroid_x"], row["centroid_y"])
            for name, point in self.entry_points.items():
                direction_to_point = (point[0] - centroid[0], point[1] - centroid[1])
                angle = _angle_between(velocity, direction_to_point)
                if angle <= self.approach_angle_threshold_deg:
                    results.append((int(row["track_id"]), name))
        return results

    def _possibly_taken_objects(self) -> Dict[int, dict]:
        flagged = {}
        obj_track_ids = self._track_spans[self._track_spans["class_name"] != "person"]["track_id"].tolist()

        for obj_id in obj_track_ids:
            obj_rows = self.df[self.df["track_id"] == obj_id].sort_values("frame_index")
            if len(obj_rows) < 10:
                continue

            nearest_person_per_frame = []
            for _, obj_row in obj_rows.iterrows():
                frame_persons = self.df[
                    (self.df["frame_index"] == obj_row["frame_index"]) & (self.df["class_name"] == "person")
                ]
                if frame_persons.empty:
                    nearest_person_per_frame.append(None)
                    continue
                dists = np.hypot(frame_persons["centroid_x"] - obj_row["centroid_x"],
                                  frame_persons["centroid_y"] - obj_row["centroid_y"])
                nearest_person_per_frame.append(int(frame_persons.loc[dists.idxmin(), "track_id"]))

            valid = [p for p in nearest_person_per_frame if p is not None]
            if not valid:
                continue
            primary_carrier = max(set(valid), key=valid.count)

            tail_window = nearest_person_per_frame[-10:]
            tail_valid = [p for p in tail_window if p is not None]
            if tail_valid and all(p != primary_carrier for p in tail_valid):
                new_owner = max(set(tail_valid), key=tail_valid.count)
                flagged[obj_id] = {
                    "class_name": obj_rows.iloc[0]["class_name"],
                    "primary_carrier": primary_carrier,
                    "new_person_nearby": new_owner,
                    "last_frame": int(obj_rows["frame_index"].max()),
                }
        return flagged

    def _altercation_pairs_at(self, frame_idx: int, frame_rows: pd.DataFrame) -> List[Tuple[int, int]]:
        """Flags pairs of people simultaneously in close, sustained contact
        with high relative motion between them -- deliberately requires TWO
        people (a lone person moving fast, e.g. a child running, cannot
        trigger this). Caveat: can still misfire on other close-contact,
        high-relative-motion activities (play-wrestling, dancing); treat as
        a candidate signal for review, not a confirmed detection."""
        person_rows = frame_rows[frame_rows["class_name"] == "person"]
        if len(person_rows) < 2:
            return []

        flagged_pairs = []
        rows = person_rows.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                dist = np.hypot(a["centroid_x"] - b["centroid_x"], a["centroid_y"] - b["centroid_y"])
                if dist > self.altercation_proximity_px:
                    continue

                rel_vx = a["velocity_x"] - b["velocity_x"]
                rel_vy = a["velocity_y"] - b["velocity_y"]
                relative_speed = np.hypot(rel_vx, rel_vy)
                if relative_speed < self.altercation_relative_speed_thresh:
                    continue

                pair_key = tuple(sorted([int(a["track_id"]), int(b["track_id"])]))
                recent_window = self.df[
                    (self.df["frame_index"] >= frame_idx - self.altercation_sustained_frames)
                    & (self.df["frame_index"] <= frame_idx)
                    & (self.df["track_id"].isin(pair_key))
                ]
                frames_both_present = recent_window.groupby("frame_index")["track_id"].nunique()
                sustained = (frames_both_present >= 2).sum() >= self.altercation_sustained_frames * 0.6

                if sustained:
                    flagged_pairs.append(pair_key)

        return flagged_pairs

    def analyze_frame(self, frame_idx: int, occupancy: str, current_time: dtime) -> FrameContext:
        frame_rows = self.df[self.df["frame_index"] == frame_idx]
        person_rows = frame_rows[frame_rows["class_name"] == "person"]
        object_rows = frame_rows[frame_rows["class_name"].isin(OBJECT_CLASSES)]
        unknown_rows = frame_rows[~frame_rows["class_name"].isin(EXPECTED_CLASSES)]

        ctx = FrameContext(
            frame_index=frame_idx,
            person_count=person_rows["track_id"].nunique(),
            object_count=object_rows["track_id"].nunique(),
            unknown_class_names=sorted(unknown_rows["class_name"].unique().tolist()),
            is_night=is_night(current_time),
            occupancy=occupancy,
        )

        if ctx.person_count == 0 and ctx.object_count == 0:
            ctx.labels.append(ContextLabel.NO_ACTIVITY)
        if ctx.person_count >= 1:
            ctx.labels.append(ContextLabel.PERSON_PRESENT)
        if ctx.person_count >= self.multiple_people_thresh:
            ctx.labels.append(ContextLabel.MULTIPLE_PEOPLE)
        if ctx.unknown_class_names:
            ctx.labels.append(ContextLabel.UNKNOWN_OBJECT)
        if ctx.is_night:
            ctx.labels.append(ContextLabel.NIGHT_ACTIVITY)

        ctx.loitering_track_ids = self._loitering_tracks_at(frame_idx)
        if ctx.loitering_track_ids:
            ctx.labels.append(ContextLabel.LOITERING)

        ctx.abandoned_track_ids = self._abandoned_objects_at(frame_idx, frame_rows)
        if ctx.abandoned_track_ids:
            ctx.labels.append(ContextLabel.ABANDONED_OBJECT)

        ctx.approaching_entry = self._approaching_entry_tracks_at(frame_rows)
        if ctx.approaching_entry:
            ctx.labels.append(ContextLabel.APPROACHING_ENTRY)

        for obj_id, info in self._possibly_taken.items():
            if abs(info["last_frame"] - frame_idx) <= 5:
                ctx.labels.append(ContextLabel.POSSIBLE_OBJECT_TAKEN)
                ctx.possibly_taken_objects.append((obj_id, info))

        ctx.altercation_pairs = self._altercation_pairs_at(frame_idx, frame_rows)
        if ctx.altercation_pairs:
            ctx.labels.append(ContextLabel.POSSIBLE_ALTERCATION)

        return ctx


@dataclass
class RiskWeights:
    mnad_score: float = 10.0
    mnad_high_thresh: float = 0.65
    mnad_score_max_boost: float = 45.0
    person_present: float = 10.0
    multiple_people: float = 8.0
    loitering: float = 15.0
    unknown_object: float = 12.0
    abandoned_object: float = 15.0
    night_activity: float = 10.0
    approaching_entry: float = 12.0
    possible_object_taken: float = 18.0
    possible_altercation: float = 30.0
    vacant_home_multiplier: float = 1.5


class RiskEngine:
    def __init__(self, weights: RiskWeights = RiskWeights()):
        self.weights = weights

    def _mnad_contribution(self, mnad_score: float) -> float:
        """Non-linear MNAD contribution: muted at low/moderate scores
        (domain-mismatch noise), sharply boosted at high scores (likely
        genuine anomaly)."""
        s = float(np.clip(mnad_score, 0.0, 1.0))
        w = self.weights
        if s >= w.mnad_high_thresh:
            span = max(1.0 - w.mnad_high_thresh, 1e-6)
            boost_fraction = (s - w.mnad_high_thresh) / span
            return w.mnad_score + boost_fraction * (w.mnad_score_max_boost - w.mnad_score)
        else:
            return w.mnad_score * (s / w.mnad_high_thresh) * 0.3

    def compute(self, mnad_score: float, ctx: FrameContext) -> float:
        w = self.weights
        score = 0.0
        score += self._mnad_contribution(mnad_score)
        if ContextLabel.PERSON_PRESENT in ctx.labels:
            score += w.person_present
        if ContextLabel.MULTIPLE_PEOPLE in ctx.labels:
            score += w.multiple_people
        if ContextLabel.LOITERING in ctx.labels:
            score += w.loitering
        if ContextLabel.UNKNOWN_OBJECT in ctx.labels:
            score += w.unknown_object
        if ContextLabel.ABANDONED_OBJECT in ctx.labels:
            score += w.abandoned_object
        if ContextLabel.NIGHT_ACTIVITY in ctx.labels:
            score += w.night_activity
        if ContextLabel.APPROACHING_ENTRY in ctx.labels:
            score += w.approaching_entry
        if ContextLabel.POSSIBLE_OBJECT_TAKEN in ctx.labels:
            score += w.possible_object_taken
        if ContextLabel.POSSIBLE_ALTERCATION in ctx.labels:
            score += w.possible_altercation
        if ctx.occupancy == "vacant":
            score *= w.vacant_home_multiplier
        return float(np.clip(score, 0.0, 100.0))


class ExplainabilityEngine:
    def explain(self, risk_score: float, ctx: FrameContext, mnad_score: float) -> List[str]:
        reasons = []
        if mnad_score >= 0.7:
            reasons.append(f"High anomaly score ({mnad_score:.2f}) from MNAD")
        elif mnad_score >= 0.4:
            reasons.append(f"Moderate anomaly score ({mnad_score:.2f}) from MNAD")
        if ContextLabel.PERSON_PRESENT in ctx.labels:
            reasons.append(f"{ctx.person_count} person(s) detected")
        if ContextLabel.MULTIPLE_PEOPLE in ctx.labels:
            reasons.append("Multiple people present simultaneously")
        if ctx.occupancy == "vacant":
            reasons.append("House marked as vacant")
        if ContextLabel.NIGHT_ACTIVITY in ctx.labels:
            reasons.append("Activity occurring during night hours")
        if ContextLabel.LOITERING in ctx.labels:
            reasons.append(f"Loitering detected (track IDs: {ctx.loitering_track_ids})")
        if ContextLabel.UNKNOWN_OBJECT in ctx.labels:
            reasons.append(f"Unrecognized object class(es): {ctx.unknown_class_names}")
        if ContextLabel.ABANDONED_OBJECT in ctx.labels:
            reasons.append(f"Possible abandoned object (track IDs: {ctx.abandoned_track_ids})")
        if ContextLabel.APPROACHING_ENTRY in ctx.labels:
            names = sorted(set(name for _, name in ctx.approaching_entry))
            reasons.append(f"Person(s) approaching: {', '.join(names)}")
        if ContextLabel.POSSIBLE_OBJECT_TAKEN in ctx.labels:
            for obj_id, info in ctx.possibly_taken_objects:
                reasons.append(
                    f"{info['class_name']} (track {obj_id}) possibly taken by person "
                    f"{info['new_person_nearby']} (originally near person {info['primary_carrier']})"
                )
        if ContextLabel.POSSIBLE_ALTERCATION in ctx.labels:
            reasons.append(f"Possible physical altercation between tracks: {ctx.altercation_pairs}")
        if not reasons:
            reasons.append("No significant risk factors detected")
        return reasons
