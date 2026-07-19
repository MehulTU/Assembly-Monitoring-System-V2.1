"""
decision/event_detector.py
================================
SUBSYSTEM: Decision (SS5) — File 5 System Architecture

This is NOT image processing. This is logic.

OpenCV (Processing) tells us: "motion detected"
Decision asks:               "does this motion mean something happened?"

STATE MACHINE:
  The system is always in one of two states:
    IDLE   → nothing is happening
    ACTIVE → assembly activity is occurring

  Transitions:
    IDLE   → ACTIVE  when motion is detected for N consecutive frames
    ACTIVE → IDLE    when no motion for N frames AND idle lasted > min duration

  This uses temporal filtering to avoid false events from brief flickers.

IMPORTANT NOTE on Pick vs Place:
  With motion detection only (V1), we CANNOT distinguish Pick from Place.
  Both look like "motion in the ROI". They are logged as "ACTIVE (Pick/Place)".
  Distinguishing Pick from Place requires object recognition (YOLO — V3).
  This is a documented V1 limitation, not a bug.
"""

import time
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    TEMPORAL_FILTER_FRAMES,
    ACTIVE_MIN_DURATION_SEC,
)


@dataclass
class AssemblyEvent:
    event_type: str
    start_time: float
    end_time: float
    duration_sec: float
    detected_object: Optional[str] = None


class EventDetector:

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"

    def __init__(self, on_event_detected=None):

        self._state = self.IDLE
        self._state_start_time = time.time()

        self._motion_streak = 0
        self._still_streak = 0

        self._on_event = on_event_detected

        self.current_state = self.IDLE
        self.current_duration = 0.0

    def update(
        self,
        motion_result,
        object_result,
    ):

        now = time.time()

        motion_detected = motion_result.motion_detected
        is_warmup = motion_result.is_warmup

        if motion_detected:
            self._motion_streak += 1
            self._still_streak = 0
        else:
            self._still_streak += 1
            self._motion_streak = 0

        completed_event = None

        detected_name = None

        if object_result.objects:
            detected_name = object_result.objects[0].class_name

        if not is_warmup:

            # IDLE -> ACTIVE

            if (
                self._state == self.IDLE
                and self._motion_streak >= TEMPORAL_FILTER_FRAMES
            ):

                self._state = self.ACTIVE
                self._state_start_time = now

            # ACTIVE -> IDLE

            elif (
                self._state == self.ACTIVE
                and self._still_streak >= TEMPORAL_FILTER_FRAMES
            ):

                duration = now - self._state_start_time

                if duration >= ACTIVE_MIN_DURATION_SEC:

                    if detected_name is None:
                        event_name = "ACTIVE"
                    else:
                        event_name = f"ACTIVE ({detected_name})"

                    completed_event = AssemblyEvent(
                        event_type=event_name,
                        start_time=self._state_start_time,
                        end_time=now,
                        duration_sec=duration,
                        detected_object=detected_name,
                    )

                    if self._on_event:
                        self._on_event(completed_event)

                self._state = self.IDLE
                self._state_start_time = now

        self.current_state = self._state
        self.current_duration = now - self._state_start_time

        return completed_event
