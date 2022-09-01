import cv2
import numpy as np
import numpy.typing as npt

from typing import Optional


__all__ = ["threshold_pixel_values"]


def threshold_pixel_values(
    frames: npt.NDArray[np.uint8],
    min_val: npt.NDArray[np.uint8],
    max_val: npt.NDArray[np.uint8],
    threshold_color_space_change: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    """
    Thresholds the pixel values. The `threshold_color_space_change` parameter corresponds to a `cv2` color space conversion, and only affects in what
    color space are the pixel thresholded, it does not change the color space of the resulting frame buffer.
    """

    def threshold_pixel_values_in_frame(frame: npt.NDArray[np.uint8]):
        frame_in_color_space = (
            frame if threshold_color_space_change is None else cv2.cvtColor(frame, threshold_color_space_change)
        )

        skin_mask = cv2.inRange(frame_in_color_space, min_val, max_val)
        return cv2.bitwise_and(frame, frame, mask=skin_mask)

    return np.array([threshold_pixel_values_in_frame(frame) for frame in frames], dtype=np.ndarray)
