"""
acquisition/camera_interface.py
================================
SUBSYSTEM: Acquisition (SS2) — File 5 System Architecture

This file defines the CameraInterface — a blueprint (abstract base class)
that ALL cameras must follow. Think of it as a contract:

  "Whatever camera you plug in, it must be able to do these things."

WHY THIS MATTERS FOR YOUR THESIS:
  Your architecture says Acquisition is independent of Processing.
  This file is what makes that true in code.

  Switching from Logitech → RealSense D415 requires changing ONLY:
    config/settings.py  →  CAMERA_TYPE = "realsense"

  Nothing in Processing, Decision, or Analytics changes at all.

This is called a Hardware Abstraction Layer (HAL).
"""

from abc import ABC, abstractmethod


class CameraInterface(ABC):
    """
    Abstract base class for all cameras.
    Every camera (Logitech, RealSense, etc.) must implement these methods.

    You will never use this class directly.
    You will use LogitechCamera or RealSenseCamera, which both follow this contract.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Open and initialise the camera.
        Returns True if successful, False if the camera could not be opened.
        """
        pass

    @abstractmethod
    def read_frame(self):
        """
        Capture one frame from the camera.

        Returns:
            (success: bool, frame: numpy array or None)
            success = True  → frame contains a valid image
            success = False → camera failed, frame is None
        """
        pass

    @abstractmethod
    def release(self):
        """
        Properly close the camera and release hardware resources.
        Always call this when done — like closing a file after reading it.
        """
        pass

    @abstractmethod
    def get_fps(self) -> float:
        """Return the actual frames-per-second the camera is running at."""
        pass

    @abstractmethod
    def get_resolution(self) -> tuple:
        """Return (width, height) of the frames this camera produces."""
        pass
