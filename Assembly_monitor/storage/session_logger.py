"""
storage/session_logger.py
================================
SUBSYSTEM: Storage (SS8) — File 5 System Architecture

Handles all data saving — CSV event logs with full session metadata.

KEY IMPROVEMENT OVER PHASE 3:
  Each CSV now includes a HEADER with trial metadata:
    - Date and time
    - Camera type and settings
    - DOE trial ID (from settings)
    - Lighting condition
    - Camera height and angle
  This means after 8 DOE trials you can tell EXACTLY which CSV
  belongs to which trial — critical for validation (File 7).
"""

import csv
import os
import time
from config.settings import (
    LOGS_DIR,
    CAMERA_TYPE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

from decision.event_detector import AssemblyEvent


class SessionLogger:
    """
    Logs assembly events to CSV with full session metadata.
    Maps to Storage subsystem (SS8) in the system architecture.
    """

    def __init__(self, trial_id: str = "T-00",
                 lighting: str = "Normal",
                 sensor_height_cm: str = "TBD",
                 sensor_angle_deg: str = "TBD"):
        """
        Args:
            trial_id:          DOE trial ID (e.g. "T-01") — from DOE planning sheet
            lighting:          Lighting condition (e.g. "Normal", "Low", "High")
            sensor_height_cm:  Camera height in cm (filled in after DOE measurement)
            sensor_angle_deg:  Camera inclination angle in degrees
        """
        self._trial_id         = trial_id
        self._lighting         = lighting
        self._sensor_height    = sensor_height_cm
        self._sensor_angle     = sensor_angle_deg
        self._session_start    = time.strftime("%Y-%m-%d %H:%M:%S")
        self._filepath         = self._build_filepath()
        self._file             = None
        self._writer           = None
        self._event_count      = 0

    def _build_filepath(self) -> str:
        """Build a descriptive filename including trial ID and timestamp."""
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"{self._trial_id}_{CAMERA_TYPE}_{timestamp}.csv"
        return os.path.join(LOGS_DIR, filename)

    def open(self):
        """Open the CSV file and write the session metadata header."""
        self._file   = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        # ── Session metadata header ────────────────────────────────────────
        # This is the KEY improvement: you can identify every CSV by its trial.
        self._writer.writerow(["# ASSEMBLY MONITORING SESSION"])
        self._writer.writerow(["# Session start",    self._session_start])
        self._writer.writerow(["# Trial ID",         self._trial_id])
        self._writer.writerow(["# Camera type",      CAMERA_TYPE])
        self._writer.writerow(["# Resolution",       f"{FRAME_WIDTH}x{FRAME_HEIGHT}"])
        self._writer.writerow(["# Lighting",         self._lighting])
        self._writer.writerow(["# Sensor height cm", self._sensor_height])
        self._writer.writerow(["# Sensor angle deg", self._sensor_angle])
        self._writer.writerow([])  # blank line before data

        # ── Column headers ─────────────────────────────────────────────────
        self._writer.writerow([
            "event_id",
            "event_type",
            "detected_object",
            "start_time",
            "end_time",
            "duration_sec",
            "trial_id",
        ])
        self._file.flush()
        print(f"[SessionLogger] Logging to: {self._filepath}")

    def log_event(self, event: AssemblyEvent):
        """Write one assembly event to the CSV.

        Called by Storage whenever Decision detects a completed event.
        """
        if self._writer is None:
            return
        self._event_count += 1
        self._writer.writerow([
            self._event_count,
            event.event_type,
            event.detected_object if event.detected_object else "",
            time.strftime("%H:%M:%S", time.localtime(event.start_time)),
            time.strftime("%H:%M:%S", time.localtime(event.end_time)),
            f"{event.duration_sec:.3f}",
            self._trial_id,
        ])
        self._file.flush()

    def close(self):
        """Close the CSV file properly."""
        if self._file:
            self._file.close()
            print(f"[SessionLogger] Session closed. {self._event_count} events logged → {self._filepath}")

    @property
    def filepath(self) -> str:
        return self._filepath
