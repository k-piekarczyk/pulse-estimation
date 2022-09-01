import cv2

from . import BaseVideoSource


__all__ = ["CameraVideoSource"]


class CameraVideoSource(BaseVideoSource):
    def __init__(self, camera_index: int):
        super().__init__(cv2.VideoCapture(camera_index))
