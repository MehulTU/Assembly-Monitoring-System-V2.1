"""
processing/motion_detector.py
================================
SUBSYSTEM: Processing (SS4) — File 5 System Architecture

Full motion detection pipeline in one clean class.
Steps inside (in order):
  1. Crop to ROI
  2. Grayscale conversion
  3. Gaussian blur (noise reduction)
  4. Background subtraction
  5. Threshold → binary mask
  6. Morphological opening (removes noise dots)
  7. Contour detection
  8. Motion score calculation (% of ROI moving)

OUTPUT:
  MotionResult object containing:
    - motion_detected (bool)
    - motion_score (float, 0–100%)
    - contours list
    - processed frame for display
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List
from config.settings import (
    BLUR_KERNEL_SIZE,
    MOTION_THRESHOLD,
    MORPH_KERNEL_SIZE,
    MIN_CONTOUR_AREA,
    MOTION_SCORE_THRESHOLD,
    BACKGROUND_WARMUP_FRAMES,
)


@dataclass
class MotionResult:
    motion_detected: bool
    motion_score: float
    contours: List = field(default_factory=list)
    bounding_boxes: List = field(default_factory=list)
    binary_mask: object = None
    is_warmup: bool = False


class MotionDetector:

    def __init__(self, roi_points=None):
        self._roi_points = roi_points
        self._roi_mask = None
        self._roi_area = None
        self._frame_count = 0

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=150,
            varThreshold=30,
            detectShadows=False,
        )

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )

    def process(self, frame):

        self._frame_count += 1
        warmup = self._frame_count <= BACKGROUND_WARMUP_FRAMES

        if self._roi_mask is None:
            self._roi_mask = self._build_roi_mask(frame.shape)
            self._roi_area = max(1, cv2.countNonZero(self._roi_mask))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(
            gray,
            (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE),
            0,
        )

        fg = self._bg_subtractor.apply(blur)

        fg = cv2.bitwise_and(
            fg,
            fg,
            mask=self._roi_mask,
        )

        _, binary = cv2.threshold(
            fg,
            MOTION_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            self._kernel,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self._kernel,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        valid = []
        boxes = []
        moving_area = 0

        for c in contours:

            area = cv2.contourArea(c)

            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)

            if w < 15 or h < 15:
                continue

            valid.append(c)
            boxes.append((x, y, w, h))
            moving_area += area

        motion_score = (moving_area / self._roi_area) * 100.0
        motion_detected = motion_score >= MOTION_SCORE_THRESHOLD

        return MotionResult(
            motion_detected=motion_detected and not warmup,
            motion_score=motion_score,
            contours=valid,
            bounding_boxes=boxes,
            binary_mask=binary,
            is_warmup=warmup,
        )

    def _build_roi_mask(self, frame_shape):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self._roi_points is None:
            mask[:, :] = 255
        else:
            cv2.fillPoly(mask, [self._roi_points], 255)
        return mask