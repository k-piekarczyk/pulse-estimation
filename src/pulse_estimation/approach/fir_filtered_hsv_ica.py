import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib.lines import Line2D
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import FastICA

from pulse_estimation.utils.video import FileVideoSource
from pulse_estimation.utils.signal import butter_bandpass_filter, fir_bandpass_filter
from pulse_estimation.core import extract_face_frames, get_mean_pixel_values, threshold_pixel_values


__all__ = ["fir_filtered_HSV_ICA"]


def fir_filtered_HSV_ICA(
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
    """ """
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
    mean_values = get_mean_pixel_values(skin_pixels, omit_zeros=True, color_space_change=cv2.COLOR_BGR2HSV)

    T = 1.0 / fps

    line_color = ["-b", "-b", "-b"]

    """
    Original signal
    """
    mv_freqs = rfftfreq(mean_values.shape[0], T)

    mv_mags = [np.abs(rfft(mean_values[:, i])) for i in range(3)]

    for i in range(3):
        mv_mags[i][0] = 0

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

    filt_freqs = rfftfreq(filtered[0].shape[0], T)
    filt_mags = [np.abs(rfft(filtered[i])) for i in range(3)]

    for i in range(3):
        filt_mags[i][0] = 0

    """
    ICA
    """
    transformer = FastICA(n_components=3, whiten="unit-variance", max_iter=1000)
    ica = transformer.fit_transform(np.swapaxes(np.array(filtered), 0, 1))

    ica_freqs = rfftfreq(ica.shape[0], T)

    ica_mags = [np.abs(rfft(ica[:, i])) for i in range(3)]

    for i in range(3):
        ica_mags[i][0] = 0

    selected_component = 0
    if np.max(ica_mags[0]) > np.max(ica_mags[1]) and np.max(ica_mags[0]) > np.max(ica_mags[2]):
        selected_component = 0
    elif np.max(ica_mags[1]) > np.max(ica_mags[2]):
        selected_component = 1
    else:
        selected_component = 2

    result = ica_freqs[np.argmax(ica_mags[selected_component])]

    """
    Plots
    """
    if plot:
        lines = [Line2D([0], [0], color="magenta", linewidth=1, linestyle="solid")]
        labels = ["Est. HR: %.2f Hz - %.2f bpm" % (result, result * 60)]
        if acc_hr is not None:
            lines.append(Line2D([0], [0], color="black", linewidth=1, linestyle="dashed"))
            labels.append("Act. HR: %.2f Hz - %.2f bpm" % (acc_hr, acc_hr * 60))

        # Signal
        fig_signal, axs_signal = plt.subplots(3, 4, constrained_layout=True)
        fig_signal.suptitle(f"ICA: Signal plots for: {target_video_file}")
        fig_signal.legend(lines, labels)

        for i in range(3):
            axs_signal[i, 0].set_title(f"Channel {i + 1}")
            axs_signal[i, 0].plot(mean_values[:, i], line_color[i])

        for i in range(3):
            axs_signal[i, 1].set_title(f"Channel {i + 1} FT")
            axs_signal[i, 1].plot(mv_freqs, mv_mags[i], line_color[i])

            if acc_hr is not None:
                axs_signal[i, 1].axvline(x=acc_hr, color="black", linestyle="dashed")
            axs_signal[i, 1].axvline(x=result, color="magenta", linestyle="solid")

        for i in range(3):
            axs_signal[i, 2].set_title(f"Channel {i + 1} filtered")
            axs_signal[i, 2].plot(filtered[i], line_color[i])

        for i in range(3):
            axs_signal[i, 3].set_title(f"Channel {i + 1} filtered FT")
            axs_signal[i, 3].plot(filt_freqs, filt_mags[i], line_color[i])

            if acc_hr is not None:
                axs_signal[i, 3].axvline(x=acc_hr, color="black", linestyle="dashed")
            axs_signal[i, 3].axvline(x=result, color="magenta", linestyle="solid")

        # Components
        fig_components, axs_components = plt.subplots(3, 2, constrained_layout=True)
        fig_components.suptitle(f"ICA: Components plots for: {target_video_file}")
        fig_components.legend(lines, labels)

        for i in range(3):
            title = f"Component {i + 1}"
            if i == selected_component:
                title += " - Selected"

            axs_components[i, 0].set_title(title)
            axs_components[i, 0].plot(ica[:, i])

        for i in range(3):
            title = f"Component {i + 1} FT"
            if i == selected_component:
                title += " - Selected"

            axs_components[i, 1].set_title(title)
            axs_components[i, 1].plot(ica_freqs, ica_mags[i])

            if acc_hr is not None:
                axs_components[i, 1].axvline(x=acc_hr, color="black", linestyle="dashed")
            axs_components[i, 1].axvline(x=result, color="magenta", linestyle="solid")

        plt.show()

    return result
