__all__ = ["NoFaceDetected"]


class NoFaceDetected(Exception):
    def __init__(self) -> None:
        super().__init__("No faces were detected in the provided source")
