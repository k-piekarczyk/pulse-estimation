import os
import cv2

from . import BaseVideoSource
from ..exceptions import NotAVideoFileException


__all__ = ["FileVideoSource"]


class FileVideoSource(BaseVideoSource):
    def __init__(self, filepath: str):
        abs_filepath = os.path.abspath(filepath)
        if not os.path.isfile(abs_filepath):
            raise NotAVideoFileException(filepath=abs_filepath)

        super().__init__(cv2.VideoCapture(abs_filepath))
