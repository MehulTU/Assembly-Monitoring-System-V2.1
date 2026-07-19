"""
main.py
================================
Vision-Based Assembly Monitoring System — V2
Main entry point.

This file wires together ALL subsystems from the architecture (File 5):

  SS2  Acquisition     → CameraFactory + CameraInterface
       Frame Validator → FrameValidator
  SS3  Calibration     → loads roi_config.json
  SS4  Processing      → MotionDetector
  SS5  Decision        → EventDetector
  SS6  Analytics       → MetricsCalculator
  SS8  Storage         → SessionLogger
  SS9  Visualization   → LiveDisplay

TO RUN:
  python main.py

TO SWITCH CAMERA:
  Edit config/settings.py  → change CAMERA_TYPE = "realsense"
  Nothing else changes.

CONTROLS:
  q  = quit
  r  = start / stop recording video
  s  = save snapshot
"""

import cv2
import json
import numpy as np
import os
import time

# ── Import all subsystems ─────────────────────────────────────────────────────
from config.settings         import (DISPLAY_SCALE, ROI_CONFIG_PATH, BACKGROUND_WARMUP_FRAMES,
                                     LOGS_DIR, RECORDINGS_DIR, FPS_TARGET,
                                     YOLO_MODEL_PATH, YOLO_CONFIDENCE,
                                     YOLO_ROLLING_WINDOW, YOLO_IMGSZ,
                                     YOLO_DEVICE,
                                     DETECTION_PRINT_INTERVAL_SEC)
from acquisition.camera_factory  import create_camera
from acquisition.frame_validator import FrameValidator
from processing.motion_detector import MotionDetector
from processing.object_detector import ObjectDetector
from decision.event_detector     import EventDetector
from analytics.metrics           import MetricsCalculator
from storage.session_logger      import SessionLogger
from visualization.display       import LiveDisplay


def load_roi(path: str):
    """Load ROI points from JSON file created by roi_selection tool."""
    if not os.path.exists(path):
        print(f"[main] WARNING: ROI config not found at '{path}'.")
        print("  → Run roi_selection.py first to define the workspace.")
        print("  → Using full frame as ROI for now.")
        return None
    with open(path) as f:
        data = json.load(f)
    points = np.array(data["points"], dtype=np.int32)
    print(f"[main] ROI loaded from: {path}")
    return points


def main():
    print("=" * 55)
    print("  Vision-Based Assembly Monitoring System — V1")
    print("=" * 55)

    # ── 1. Load ROI (Calibration / SS3) ──────────────────────────────────
    roi_points = load_roi(ROI_CONFIG_PATH)

    # ── 2. Open camera (Acquisition / SS2) ───────────────────────────────
    camera = create_camera()
    if not camera.open():
        print("[main] FATAL: Could not open camera. Exiting.")
        return

    fps = camera.get_fps() or FPS_TARGET

    # ── 3. Initialise all subsystems ──────────────────────────────────────
    validator  = FrameValidator()
    motion_detector = MotionDetector(roi_points=roi_points)

    object_detector = ObjectDetector(
        model_path=YOLO_MODEL_PATH,
        confidence=YOLO_CONFIDENCE,
        rolling_window=YOLO_ROLLING_WINDOW,
        imgsz=YOLO_IMGSZ,
        device=YOLO_DEVICE,
    )
    logger     = SessionLogger(
        trial_id         = "T-01",      # change for each DOE trial
        lighting         = "Normal",    # change per DOE trial
        sensor_height_cm = "TBD",       # fill after lab measurement
        sensor_angle_deg = "TBD",       # fill after lab measurement
    )
    analytics  = MetricsCalculator()
    display    = LiveDisplay(roi_points=roi_points)

    def on_event(event):
        """Called every time Decision detects a completed event."""
        logger.log_event(event)
        analytics.process_event(event)
        print(f"[EVENT] {event.event_type:25s}  duration={event.duration_sec:.2f}s")

    events     = EventDetector(on_event_detected=on_event)
    logger.open()

    # ── 4. Video writer (optional recording) ─────────────────────────────
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    recording  = False
    writer     = None
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    w, h       = camera.get_resolution()

    print(f"\n  Controls: [q] quit   [r] record/stop   [s] snapshot")
    print(f"  Warming up background model ({BACKGROUND_WARMUP_FRAMES} frames)...\n")

    # Throttle for terminal detection summaries (avoids flooding the
    # terminal with one line per detection per frame).
    last_detection_print = 0.0

    # ── 5. Main loop ──────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("[main] Camera read failed. Stopping.")
                break

            # Frame Validation (between SS2 and SS4)
            valid, reason = validator.validate(frame)
            if not valid:
                print(f"[FrameValidator] Skipped frame: {reason}")
                continue

            # Processing (SS4)
            motion_result = motion_detector.process(frame)
            object_result = object_detector.process(frame)

            # Throttled per-class confidence summary in the terminal.
            # Example:  [YOLO] marker 43% (avg 38%, x2) | power_adapter 61% (avg 55%, x1)
            now = time.time()
            if (
                object_result.class_stats
                and now - last_detection_print >= DETECTION_PRINT_INTERVAL_SEC
            ):
                summary = " | ".join(
                    f"{s.class_name} {s.best_confidence * 100:.0f}% "
                    f"(avg {s.rolling_confidence * 100:.0f}%, x{s.count})"
                    for s in object_result.class_stats.values()
                )
                print(f"[YOLO] {summary}")
                last_detection_print = now

            # Decision (SS5)
            # detector was undefined; use motion_detector's frame count (fallback to 0)
            warmup_remaining = max(
                0,
                BACKGROUND_WARMUP_FRAMES - motion_detector._frame_count
            )

            completed_event = events.update(
                motion_result=motion_result,
                object_result=object_result,
            )

            # Visualization (SS9)
            metrics_summary = analytics.get_summary()

            display_frame = display.render(
                frame=frame,
                motion_result=motion_result,
                object_result=object_result,
                event_state=events.current_state,
                event_duration=events.current_duration,
                metrics_summary=metrics_summary,
                is_warmup=motion_result.is_warmup,
                warmup_remaining=warmup_remaining,
            )

            cv2.imshow("Assembly Monitoring System — V1", display_frame)
            cv2.imshow("Motion Mask (debug)", motion_result.binary_mask)

            if DISPLAY_SCALE != 1.0:
                display_frame = cv2.resize(
                    display_frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )

            if recording and writer:
                writer.write(frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    fname   = os.path.join(RECORDINGS_DIR, f"recording_{int(time.time())}.mp4")
                    writer  = cv2.VideoWriter(fname, fourcc, fps, (w, h))
                    recording = True
                    print(f"[main] Recording started: {fname}")
                else:
                    recording = False
                    if writer: writer.release(); writer = None
                    print("[main] Recording stopped.")
            elif key == ord('s'):
                fname = os.path.join(RECORDINGS_DIR, f"snapshot_{int(time.time())}.png")
                cv2.imwrite(fname, frame)
                print(f"[main] Snapshot saved: {fname}")

    finally:
        if writer: writer.release()
        logger.close()
        camera.release()
        cv2.destroyAllWindows()
        print("\n[main] System stopped cleanly.")
        print(f"[main] Final metrics: {analytics.get_summary()}")


if __name__ == "__main__":
    main()
