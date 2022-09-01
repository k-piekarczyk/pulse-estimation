__all__ = ["ClosedVideoSourceException"]


class ClosedVideoSourceException(Exception):
    def __init__(self) -> None:
        super().__init__("The VideoSource you're trying to access is closed")
