__all__ = ["VideoCaptureClosedException"]


class VideoCaptureClosedException(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't open video capture")
