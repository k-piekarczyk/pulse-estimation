import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.decomposition import FastICA
from scipy.fft import rfft, rfftfreq

from pulse_estimation.utils.video import FileVideoSource
from pulse_estimation.utils.signal import butter_bandpass_filter
from pulse_estimation.core import extract_face_frames, get_mean_pixel_values, threshold_pixel_values

face_cascade_file = "./resources/haar/haarcascade_frontalface_default.xml"
target_video_file = "./resources/video/my/face_webcam_uncontrolled.mp4"  # Naiwna metoda wyboru komponentu NIE działa
# target_video_file = "./resources/video/my/face_controlled_not_diffused.mov" # Naiwna metoda wyboru komponentu NIE działa
# target_video_file = "./resources/video/my/face_controlled_diffused.mov" # Naiwna metoda wyboru komponentu działa
# target_video_file = "./resources/video/other/face.mp4"  # Naiwna metoda wyboru komponentu działa

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

hr_low = 0.5
hr_high = 3

def main():

    video_source = FileVideoSource(filepath=target_video_file)

    _, _, _, fps = video_source.get_stats()

    face_frames = extract_face_frames(vid=video_source, face_cascade_path=face_cascade_file, display=True)
    skin_pixels = threshold_pixel_values(frames=face_frames, min_val=min_YCrCb, max_val=max_YCrCb, threshold_color_space_change=cv2.COLOR_BGR2YCrCb)
    mean_values = get_mean_pixel_values(skin_pixels, omit_zeros=True)

    transformer = FastICA(n_components=3)

    ica = transformer.fit_transform(mean_values)

    N = ica.shape[0]
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
    axs[0, 1].plot(ica[:, 0])

    axs[1, 1].set_title("2nd component")
    axs[1, 1].plot(ica[:, 1])

    axs[2, 1].set_title("3rd component")
    axs[2, 1].plot(ica[:, 2])


    ica_comp_1_filt = butter_bandpass_filter(data=ica[:, 0], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)
    ica_comp_2_filt = butter_bandpass_filter(data=ica[:, 1], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)
    ica_comp_3_filt = butter_bandpass_filter(data=ica[:, 2], lowcut=hr_low, highcut=hr_high, sampling_rate=fps, order=5)
    
    axs[0, 2].set_title("1st component - filtered")
    axs[0, 2].plot(ica_comp_1_filt)

    axs[1, 2].set_title("2nd component - filtered")
    axs[1, 2].plot(ica_comp_2_filt)

    axs[2, 2].set_title("3rd component - filtered")
    axs[2, 2].plot(ica_comp_3_filt)

    
    ica_fft_comp_1 = np.abs(rfft(ica_comp_1_filt))
    ica_fft_comp_2 = np.abs(rfft(ica_comp_2_filt))
    ica_fft_comp_3 = np.abs(rfft(ica_comp_3_filt))

    max_freq_comp_1 = frequencies[np.argmax(ica_fft_comp_1)]
    max_freq_comp_2 = frequencies[np.argmax(ica_fft_comp_2)]
    max_freq_comp_3 = frequencies[np.argmax(ica_fft_comp_3)]

    axs[0, 3].set_title("1st component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_1, max_freq_comp_1 * 60))
    axs[0, 3].plot(frequencies, ica_fft_comp_1)

    axs[1, 3].set_title("2nd component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_2, max_freq_comp_2 * 60))
    axs[1, 3].plot(frequencies, ica_fft_comp_2)

    axs[2, 3].set_title("3rd component FFT | max: [%.4f Hz - %.2f bpm]" % (max_freq_comp_3, max_freq_comp_3 * 60))
    axs[2, 3].plot(frequencies, ica_fft_comp_3)

    comp_1_magnitude_ratio = np.partition(ica_fft_comp_1.flatten(), -2)[-2] / np.max(ica_fft_comp_1)
    comp_2_magnitude_ratio = np.partition(ica_fft_comp_2.flatten(), -2)[-2] / np.max(ica_fft_comp_2)
    comp_3_magnitude_ratio = np.partition(ica_fft_comp_3.flatten(), -2)[-2] / np.max(ica_fft_comp_3)

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

    
