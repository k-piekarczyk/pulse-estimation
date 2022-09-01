import cv2
import numpy as np
import numpy.typing as npt

from typing import Optional


__all__ = ["get_mean_pixel_values"]


def get_mean_pixel_values(
    frames: npt.NDArray[np.uint8], color_space_change: Optional[int] = None, omit_zeros: bool = False
) -> npt.NDArray[np.uint8]:
    """
    Returns mean pixel values for each frame, in a given color space.

    If `omit_zeros` is `True`, cells of value `0` will be ignored. Used to get mean values in a masked frame.
    """

    def get_mean_pixel_values_from_frame(frame: npt.NDArray[np.uint8]):
        frame_in_color_space = frame if color_space_change is None else cv2.cvtColor(frame, color_space_change)

        if omit_zeros:
            return np.nanmean(np.nanmean(np.where(frame_in_color_space != 0, frame_in_color_space, np.nan), 1), 0)
        else:
            return np.mean(np.mean(frame_in_color_space, 1), 0)

    return np.array([get_mean_pixel_values_from_frame(frame) for frame in frames])
