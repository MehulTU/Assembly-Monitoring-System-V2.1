"""
acquisition/realsense_camera.py
================================
SUBSYSTEM: Acquisition (SS2) — Intel RealSense D415 implementation

This is the future V2 camera implementation for the RealSense D415.
It follows the same CameraInterface contract as LogitechCamera.

STATUS: Stub — ready to complete once D415 is available at PLCM lab.

TO ACTIVATE D415:
  1. Install SDK:  pip install pyrealsense2
  2. Change in config/settings.py:  CAMERA_TYPE = "realsense"
  3. Uncomment the code below (remove the triple-quote blocks)

Everything else in the system (Processing, Decision, Analytics) stays IDENTICAL.
This is the power of the Hardware Abstraction Layer.
"""

from acquisition.camera_interface import CameraInterface
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET


class RealSenseCamera(CameraInterface):
    """
    Intel RealSense D415 using pyrealsense2.
    RGB stream used in V1. Depth stream available for future extensions.
    """

    def __init__(self):
        self._pipeline = None
        self._config   = None
        self._width    = FRAME_WIDTH
        self._height   = FRAME_HEIGHT
        self._fps      = FPS_TARGET

    def open(self) -> bool:
        """
        Initialise the RealSense D415 pipeline.
        Uncomment when pyrealsense2 is installed and D415 is connected.
        """
        try:
            import pyrealsense2 as rs

            self._pipeline = rs.pipeline()
            self._config   = rs.config()

            # Enable RGB colour stream
            self._config.enable_stream(
                rs.stream.color,
                self._width, self._height,
                rs.format.bgr8,
                int(self._fps)
            )

            # Uncomment to also enable depth stream (future extension):
            # self._config.enable_stream(rs.stream.depth,
            #     self._width, self._height, rs.format.z16, int(self._fps))

            self._pipeline.start(self._config)
            # Allow camera auto-exposure to stabilise
            for _ in range(30):
                self._pipeline.wait_for_frames()
            print(f"[RealSenseCamera] D415 opened: {self._width}x{self._height} @ {self._fps:.0f} fps")
            print("[RealSenseCamera] RGB stream active. Depth available for future use.")
            return True

        except ImportError:
            print("[RealSenseCamera] ERROR: pyrealsense2 not installed.")
            print("  → Run:  pip install pyrealsense2")
            return False
        except Exception as e:
            print(f"[RealSenseCamera] ERROR: Could not open D415: {e}")
            return False

    def read_frame(self):
        """Read one RGB frame from the D415."""
        if self._pipeline is None:
            return False, None
        try:
            import pyrealsense2 as rs
            frames       = self._pipeline.wait_for_frames()
            colour_frame = frames.get_color_frame()
            if not colour_frame:
                return False, None
            import numpy as np
            frame = np.asanyarray(colour_frame.get_data())
            return True, frame
        except Exception as e:
            print(f"[RealSenseCamera] Frame read error: {e}")
            return False, None

    def release(self):
        """Stop the RealSense pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            print("[RealSenseCamera] D415 pipeline stopped.")

    def get_fps(self) -> float:
        return self._fps

    def get_resolution(self) -> tuple:
        return (self._width, self._height)
