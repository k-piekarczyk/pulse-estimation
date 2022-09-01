__all__ = ["NotAVideoFileException"]


class NotAVideoFileException(Exception):
    def __init__(self, filepath: str) -> None:
        super().__init__(f"Provided file '{filepath}' is not a video file")
