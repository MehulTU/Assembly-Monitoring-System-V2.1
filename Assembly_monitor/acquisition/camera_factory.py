"""
acquisition/camera_factory.py
================================
SUBSYSTEM: Acquisition (SS2) — Camera Factory

This file reads the CAMERA_TYPE from settings.py and returns
the correct camera object. No other part of the system needs
to know which camera is being used.

USAGE (in main.py):
    from acquisition.camera_factory import create_camera
    camera = create_camera()   # automatically picks Logitech or D415
    camera.open()
    ret, frame = camera.read_frame()

TO SWITCH CAMERAS:
    Open config/settings.py
    Change:  CAMERA_TYPE = "logitech"
         to: CAMERA_TYPE = "realsense"
    Done. Nothing else changes.
"""

from config.settings import CAMERA_TYPE
from acquisition.camera_interface import CameraInterface


def create_camera() -> CameraInterface:
    """
    Factory function — returns the correct camera object based on settings.

    Returns:
        A camera object that implements CameraInterface.

    Raises:
        ValueError if an unknown CAMERA_TYPE is set in settings.py.
    """

    if CAMERA_TYPE == "logitech":
        from acquisition.logitech_camera import LogitechCamera
        print(f"[CameraFactory] Creating Logitech webcam (CAMERA_TYPE='{CAMERA_TYPE}')")
        return LogitechCamera()

    elif CAMERA_TYPE == "realsense":
        from acquisition.realsense_camera import RealSenseCamera
        print(f"[CameraFactory] Creating RealSense D415 (CAMERA_TYPE='{CAMERA_TYPE}')")
        return RealSenseCamera()

    else:
        raise ValueError(
            f"[CameraFactory] Unknown CAMERA_TYPE: '{CAMERA_TYPE}'. "
            f"Valid options: 'logitech', 'realsense'. "
            f"Check config/settings.py."
        )
