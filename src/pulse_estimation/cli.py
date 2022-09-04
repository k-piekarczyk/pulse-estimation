import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.decomposition import FastICA
from scipy.fft import rfft, rfftfreq

from scipy.signal import butter, lfilter

from pulse_estimation.utils.video import FileVideoSource
from pulse_estimation.core import extract_face_frames, get_mean_pixel_values, threshold_pixel_values

face_cascade_file = "./resources/haar/haarcascade_frontalface_default.xml"
target_video_file = "./resources/video/my/face_controlled_not_diffused.mov"

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)


def main():

    video_source = FileVideoSource(filepath=target_video_file)

    _, _, _, fps = video_source.get_stats()

    face_frames = extract_face_frames(vid=video_source, face_cascade_path=face_cascade_file, display=True)
    skin_pixels = threshold_pixel_values(frames=face_frames, min_val=min_YCrCb, max_val=max_YCrCb, threshold_color_space_change=cv2.COLOR_BGR2YCrCb)
    mean_values = get_mean_pixel_values(skin_pixels, omit_zeros=True)

    transformer = FastICA(n_components=2)

    ica = transformer.fit_transform(mean_values[:, 1:3])

    N = ica.shape[0]
    T = 1.0 / fps

    frequencies = rfftfreq(N, T)

    # ica_fft_comp_1 = np.abs(rfft(ica[:, 0]))
    # ica_fft_comp_2 = np.abs(rfft(ica[:, 1]))
    # ica_fft_comp_3 = np.abs(rfft(ica[:, 2]))

    _, axs = plt.subplots(3, 4)

    axs[0, 0].set_title("1st channel")
    axs[0, 0].plot(mean_values[:, 0], "-b")

    axs[1, 0].set_title("2nd channel")
    axs[1, 0].plot(mean_values[:, 1], "-g")

    axs[2, 0].set_title("3rd channel")
    axs[2, 0].plot(mean_values[:, 2], "-r")

    axs[0, 1].set_title("1st component")
    axs[0, 1].plot(ica[:, 0])

    axs[1, 1].set_title("2nd component")
    axs[1, 1].plot(ica[:, 1])

    # axs[2, 1].set_title("3rd component")
    # axs[2, 1].plot(ica[:, 2])

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    low_cut = 0.5
    high_cut = 3

    ica_comp_1_filt = butter_bandpass_filter(ica[:, 0], low_cut, high_cut, fps, order=5)
    ica_comp_2_filt = butter_bandpass_filter(ica[:, 1], low_cut, high_cut, fps, order=5)
    
    axs[0, 2].set_title("1st component - filtered")
    axs[0, 2].plot(ica_comp_1_filt)

    axs[1, 2].set_title("2nd component - filtered")
    axs[1, 2].plot(ica_comp_2_filt)

    
    ica_fft_comp_1 = np.abs(rfft(ica_comp_1_filt))
    ica_fft_comp_2 = np.abs(rfft(ica_comp_2_filt))


    axs[0, 3].set_title("1st component FFT")
    axs[0, 3].plot(frequencies, ica_fft_comp_1)

    axs[1, 3].set_title("2nd component FFT")
    axs[1, 3].plot(frequencies, ica_fft_comp_2)

    # axs[2, 2].set_title("3rd component FFT")
    # axs[2, 2].plot(frequencies, ica_fft_comp_3)

    plt.show()

    # max_freq_comp_1 = frequencies[np.argmax(ica_fft_comp_1)]
    # print(max_freq_comp_1, max_freq_comp_1 * 60)

    # max_freq_comp_2 = frequencies[np.argmax(ica_fft_comp_2)]
    # print(max_freq_comp_2, max_freq_comp_2 * 60)

    # max_freq_comp_3 = frequencies[np.argmax(ica_fft_comp_3)]
    # print(max_freq_comp_3, max_freq_comp_3 * 60)
