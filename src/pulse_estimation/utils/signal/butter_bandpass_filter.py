import numpy.typing as npt

from scipy.signal import butter, lfilter


__all__ = ["butter_bandpass_filter"]


def butter_bandpass_filter(data: npt.NDArray, lowcut: float, highcut: float, sampling_rate: float, order: int = 5):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y
