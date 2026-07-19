"""
acquisition/logitech_camera.py
================================
SUBSYSTEM: Acquisition (SS2) — Logitech Webcam implementation

This is the V1 camera implementation using the Logitech webcam.
It follows the CameraInterface contract defined in camera_interface.py.

WHEN YOU SWITCH TO D415:
  You will NOT touch this file.
  You will add realsense_camera.py alongside this file.
  camera_factory.py will handle which one to load.
"""

import cv2
from acquisition.camera_interface import CameraInterface
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET


class LogitechCamera(CameraInterface):
    """
    Logitech webcam using OpenCV VideoCapture.
    This is the V1 camera for initial development and testing.
    """

    def __init__(self):
        self._cap = None          # OpenCV VideoCapture object
        self._width  = FRAME_WIDTH
        self._height = FRAME_HEIGHT
        self._fps    = FPS_TARGET

    def open(self) -> bool:
        """Open the Logitech webcam."""
        self._cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self._cap.isOpened():
            print(f"[LogitechCamera] ERROR: Could not open camera at index {CAMERA_INDEX}.")
            print("  → Try changing CAMERA_INDEX in config/settings.py to 1 or 2.")
            return False

        # Set resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

        # Read back actual values (camera may not support exact values requested)
        self._width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps    = self._cap.get(cv2.CAP_PROP_FPS) or FPS_TARGET

        print(f"[LogitechCamera] Opened successfully: {self._width}x{self._height} @ {self._fps:.0f} fps")
        return True

    def read_frame(self):
        """Read one frame from the webcam."""
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            print("[LogitechCamera] Camera released.")

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> tuple:
        return (self._width, self._height)
