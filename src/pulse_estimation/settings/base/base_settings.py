from abc import ABC
from dataclasses import dataclass


__all__ = ["BaseSettings"]


@dataclass
class BaseSettings(ABC):
    face_haar_cascade_path: str
