"""
acquisition/frame_validator.py
================================
SUBSYSTEM: Frame Validation (between SS2 Acquisition and SS4 Processing)

This subsystem was identified as MISSING from GPT's pipeline.
In industry, you never blindly process a frame from the camera
without first checking it is valid.

WHAT THIS CHECKS:
  1. Is the frame empty (camera disconnected)?
  2. Has brightness suddenly changed (light switched on/off → false motion)?
  3. Is the frame too blurry to process reliably?

WHY THIS MATTERS FOR YOUR THESIS:
  Without this, a sudden lighting change produces a massive false motion event.
  Your accuracy metrics (File 7 Sheet 4) would be damaged by invalid frames.
  This directly protects your > 80% accuracy and < 15% false event criteria.
"""

import cv2
import numpy as np
from config.settings import MAX_BRIGHTNESS_JUMP, MIN_SHARPNESS_SCORE


class FrameValidator:
    """
    Validates each frame before it enters the processing pipeline.
    Keeps track of the previous frame's brightness to detect sudden changes.
    """

    def __init__(self):
        self._prev_brightness = None   # brightness of the last valid frame
        self.invalid_frame_count = 0   # how many frames have been rejected

    def validate(self, frame) -> tuple:
        """
        Check whether a frame is valid for processing.

        Args:
            frame: numpy array from camera.read_frame()

        Returns:
            (is_valid: bool, reason: str)
            is_valid = True  → frame is good, pass it to Processing
            is_valid = False → frame is bad, skip it and log the reason
        """

        # Check 1: Frame must not be None or empty
        if frame is None or frame.size == 0:
            self.invalid_frame_count += 1
            return False, "Empty frame — camera may be disconnected"

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check 2: Sudden brightness change (lighting event)
        current_brightness = float(np.mean(gray))
        if self._prev_brightness is not None:
            brightness_change = abs(current_brightness - self._prev_brightness)
            if brightness_change > MAX_BRIGHTNESS_JUMP:
                self._prev_brightness = current_brightness
                self.invalid_frame_count += 1
                return False, f"Sudden brightness change: {brightness_change:.1f} (threshold: {MAX_BRIGHTNESS_JUMP})"
        self._prev_brightness = current_brightness

        # Check 3: Sharpness — blurry frames give unreliable contours
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < MIN_SHARPNESS_SCORE:
            self.invalid_frame_count += 1
            return False, f"Frame too blurry: sharpness={laplacian_var:.1f} (min: {MIN_SHARPNESS_SCORE})"

        return True, "OK"

    def reset(self):
        """Reset brightness history — call this if camera is moved or relaunched."""
        self._prev_brightness = None
        self.invalid_frame_count = 0
