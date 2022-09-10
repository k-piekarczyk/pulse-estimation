import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.fft import rfft, rfftfreq

from pulse_estimation.utils.video import FileVideoSource
from pulse_estimation.utils.signal import butter_bandpass_filter
from pulse_estimation.core import extract_face_frames, get_mean_pixel_values, threshold_pixel_values


__all__ = ["naive_PCA"]


def naive_PCA(
    target_video_file: str,
    face_cascade_file: str,
    hr_low: float,
    hr_high: float,
    skin_min_threshold: npt.NDArray,
    skin_max_threshold: npt.NDArray,
    display: bool = False,
):
    video_source = FileVideoSource(filepath=target_video_file)

    _, _, _, fps = video_source.get_stats()

    face_frames = extract_face_frames(vid=video_source, face_cascade_path=face_cascade_file, display=display)
    skin_pixels = threshold_pixel_values(
        frames=face_frames,
        min_val=skin_min_threshold,
        max_val=skin_max_threshold,
        threshold_color_space_change=cv2.COLOR_BGR2YCrCb,
    )
    mean_values = get_mean_pixel_values(skin_pixels, omit_zeros=True)

    transformer = PCA(n_components=3)

    pca = transformer.fit_transform(mean_values)

    N = pca.shape[0]
    T = 1.0 / fps

    frequencies = rfftfreq(N, T)

    fig, axs = plt.subplots(3, 4, constrained_layout=True)

    axs[0, 0].set_title("1st channel")
    axs[0, 0].plot(mean_values[:, 0], "-b")

    axs[1, 0].set_title("2nd channel")
    axs[1, 0].plot(mean_values[:, 1], "-g")

    axs[2, 0].set_title("3rd channel")
    axs[2, 0].plot(mean_values[:, 2], "-r")

    axs[0, 1].set_title("1st component")
    axs[0, 1].plot(pca[:, 0])

    axs[1, 1].set_title("2nd component")
    axs[1, 1].plot(pca[:, 1])

    axs[2, 1].set_title("3rd component")
    axs[2, 1].plot(pca[:, 2])

    pca_comp_1_filt = butter_bandpass_filter(data=pca[:, 0], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)
    pca_comp_2_filt = butter_bandpass_filter(data=pca[:, 1], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)
    pca_comp_3_filt = butter_bandpass_filter(data=pca[:, 2], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)

    axs[0, 2].set_title("1st component - filtered")
    axs[0, 2].plot(pca_comp_1_filt)

    axs[1, 2].set_title("2nd component - filtered")
    axs[1, 2].plot(pca_comp_2_filt)

    axs[2, 2].set_title("3rd component - filtered")
    axs[2, 2].plot(pca_comp_3_filt)

    pca_fft_comp_1 = np.abs(rfft(pca_comp_1_filt))
    pca_fft_comp_2 = np.abs(rfft(pca_comp_2_filt))
    pca_fft_comp_3 = np.abs(rfft(pca_comp_3_filt))

    max_freq_comp_1 = frequencies[np.argmax(pca_fft_comp_1)]
    max_freq_comp_2 = frequencies[np.argmax(pca_fft_comp_2)]
    max_freq_comp_3 = frequencies[np.argmax(pca_fft_comp_3)]

    axs[0, 3].set_title("1st component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_1, max_freq_comp_1 * 60))
    axs[0, 3].plot(frequencies, pca_fft_comp_1)

    axs[1, 3].set_title("2nd component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_2, max_freq_comp_2 * 60))
    axs[1, 3].plot(frequencies, pca_fft_comp_2)

    axs[2, 3].set_title("3rd component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_3, max_freq_comp_3 * 60))
    axs[2, 3].plot(frequencies, pca_fft_comp_3)

    comp_1_magnitude_ratio = np.partition(pca_fft_comp_1.flatten(), -2)[-2] / np.max(pca_fft_comp_1)
    comp_2_magnitude_ratio = np.partition(pca_fft_comp_2.flatten(), -2)[-2] / np.max(pca_fft_comp_2)
    comp_3_magnitude_ratio = np.partition(pca_fft_comp_3.flatten(), -2)[-2] / np.max(pca_fft_comp_3)

    estimated_heart_rate = 0.0
    if comp_1_magnitude_ratio < comp_2_magnitude_ratio and comp_1_magnitude_ratio < comp_3_magnitude_ratio:
        estimated_heart_rate = max_freq_comp_1 * 60
    elif comp_2_magnitude_ratio < comp_3_magnitude_ratio:
        estimated_heart_rate = max_freq_comp_2 * 60
    else:
        estimated_heart_rate = max_freq_comp_3 * 60

    print("Estimated heart rate is %.2f" % (estimated_heart_rate))
    fig.suptitle("Estimated heart rate is %.2f" % (estimated_heart_rate))

    plt.show()
