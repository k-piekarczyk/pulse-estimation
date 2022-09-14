import numpy.typing as npt

from scipy.signal import firwin, lfilter


__all__ = ["fir_bandpass_filter"]


def fir_bandpass_filter(data: npt.NDArray, lowcut: float, highcut: float, sampling_rate: float, numtaps: int = 10):
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    b = firwin(numtaps, [low, high], pass_zero=False)
    y = lfilter(b, [1.0], data)
    return y
