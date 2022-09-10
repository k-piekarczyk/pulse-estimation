import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.decomposition import FastICA
from scipy.fft import rfft, rfftfreq

from pulse_estimation.utils.video import FileVideoSource
from pulse_estimation.utils.signal import butter_bandpass_filter, fir_bandpass_filter
from pulse_estimation.core import extract_face_frames, get_mean_pixel_values, threshold_pixel_values


__all__ = ["fir_filtered_RG_ICA"]


def fir_filtered_RG_ICA(
    target_video_file: str,
    face_cascade_file: str,
    hr_low: float,
    hr_high: float,
    skin_min_threshold: npt.NDArray,
    skin_max_threshold: npt.NDArray,
    acc_hr: Optional[float] = None,
    display_face_selection: bool = False,
    plot: bool = False,
) -> float:
    """
    This method consist of the following steps:

    1. Filter the channels with a FIR bandpass filter
    2. Run a 2 component ICA on the red and green channel (when there is less blood under the skin, it has a more pale, yellowy hue, and yellow is mainly a mix of red and green)
    3. Select the frequency with the highest magnitude
    """
    video_source = FileVideoSource(filepath=target_video_file)

    _, _, _, fps = video_source.get_stats()

    face_frames = extract_face_frames(
        vid=video_source, face_cascade_path=face_cascade_file, display=display_face_selection
    )
    skin_pixels = threshold_pixel_values(
        frames=face_frames,
        min_val=skin_min_threshold,
        max_val=skin_max_threshold,
        threshold_color_space_change=cv2.COLOR_BGR2YCrCb,
    )
    mean_values = get_mean_pixel_values(skin_pixels, omit_zeros=True)

    T = 1.0 / fps

    fig, axs = plt.subplots(3, 4, constrained_layout=True)

    line_color = ["-b", "-g", "-r"]

    """
    Original signal
    """
    if plot:
        for i in range(3):
            axs[i, 0].set_title(f"Channel {i + 1}")
            axs[i, 0].plot(mean_values[:, i], line_color[i])

    mv_freqs = rfftfreq(mean_values.shape[0], T)

    mv_mags = [np.abs(rfft(mean_values[:, i])) for i in range(3)]

    for i in range(3):
        mv_mags[i][0] = 0

    if plot:
        for i in range(3):
            axs[i, 1].set_title(f"Channel {i + 1} FT")
            axs[i, 1].plot(mv_freqs, mv_mags[i], line_color[i])
            if acc_hr is not None:
                axs[i, 1].axvline(x=acc_hr, color="magenta", linestyle="dashed")

    """
    Filtered signal
    """
    fir_order = 71
    filtered = [
        fir_bandpass_filter(
            data=mean_values[:, i], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, numtaps=fir_order
        )[fir_order:]
        for i in range(3)
    ]

    if plot:
        for i in range(3):
            axs[i, 2].set_title(f"Channel {i + 1} filtered")
            axs[i, 2].plot(filtered[i], line_color[i])

    filt_freqs = rfftfreq(filtered[0].shape[0], T)

    filt_mags = [np.abs(rfft(filtered[i])) for i in range(3)]

    for i in range(3):
        filt_mags[i][0] = 0

    if plot:
        for i in range(3):
            axs[i, 3].set_title(f"Channel {i + 1} filtered FT")
            axs[i, 3].plot(filt_freqs, filt_mags[i], line_color[i])
            if acc_hr is not None:
                axs[i, 3].axvline(x=acc_hr, color="magenta", linestyle="dashed")

    if plot:
        plt.show()

    """
    ICA
    """
    transformer = FastICA(n_components=2, whiten="unit-variance", max_iter=1000)
    ica = transformer.fit_transform(np.swapaxes(np.array(filtered)[1:3], 0, 1))

    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    if plot:
        for i in range(2):
            axs[i, 0].set_title(f"Component {i + 1}")
            axs[i, 0].plot(ica[:, i])

    ica_freqs = rfftfreq(ica.shape[0], T)

    ica_mags = [np.abs(rfft(ica[:, i])) for i in range(2)]

    for i in range(2):
        ica_mags[i][0] = 0

    if plot:
        for i in range(2):
            axs[i, 1].set_title(f"Component {i + 1} FT")
            axs[i, 1].plot(ica_freqs, ica_mags[i])
            if acc_hr is not None:
                axs[i, 1].axvline(x=acc_hr, color="magenta", linestyle="dashed")

    if plot:
        plt.show()

    if np.max(ica_mags[0]) > np.max(ica_mags[1]):
        return ica_freqs[np.argmax(ica_mags[0])]
    else:
        return ica_freqs[np.argmax(ica_mags[1])]
