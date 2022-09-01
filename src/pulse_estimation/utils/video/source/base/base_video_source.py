from abc import ABC
from typing import Tuple

import cv2

from ...exceptions import VideoCaptureClosedException, ClosedVideoSourceException


__all__ = ["BaseVideoSource"]


class BaseVideoSource(ABC):
    def __init__(self, cap):
        super().__init__()

        # Check if video capture opened successfully
        if cap.isOpened() is False:
            raise VideoCaptureClosedException()

        self.cap = cap
        self.open = True

    def close(self):
        """
        Closes the video source
        """
        if not self.open:
            raise ClosedVideoSourceException()

        self.open = False
        self.cap.release()

    def get_cap(self):
        """
        Returns raw cv2.VideoCapture, closes the VideoFileReader
        """
        if not self.open:
            raise ClosedVideoSourceException()

        self.open = False
        return self.cap

    def get_stats(self) -> Tuple[int, int, int, float]:
        """
        Returns a tuple with video statistics: `(width, height, frame_count, fps)`
        """
        if not self.open:
            raise ClosedVideoSourceException()
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        return height, width, frame_count, fps
