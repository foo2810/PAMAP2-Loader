from typing import Sequence
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal

def lpf(y:np.ndarray, fpass:int, fs:int) -> np.ndarray:
    """low pass filter

    Parameters
    ----------
    y: np.ndarray
        source data
    fpass: float
        catoff frequency
    fs: int
        sampling frequency

    Returns
    -------
    np.ndarray:
        filtered data
    """
    yf = fft(y)
    yf /= (len(y)/2.)
    yf[0] = yf[0] / 2.

    freq = np.linspace(0, fs, len(y))
    yf[freq > fpass] = 0

    yd = ifft(yf)
    yd = np.real(yd * len(y))

    return yd

def hpf(y:np.ndarray, fpass:int, fs:int) -> np.ndarray:
    """high pass filter

    Parameters
    ----------
    y: np.ndarray
        source data
    fpass: float
        catoff frequency
    fs: int
        sampling frequency

    Returns
    -------
    np.ndarray:
        filtered data
    """
    yf = fft(y)
    yf /= (len(y)/2.)
    yf[0] = yf[0] / 2.

    freq = np.linspace(0, fs, len(y))
    yf[freq < fpass] = 0

    yd = ifft(yf)
    yd = np.real(yd * len(y))

    return yd



def bpf(y:np.ndarray, fpass:Sequence[int, int], fs:int) -> np.ndarray:
    """filter function on data

    Parameters
    ----------
    y: np.ndarray
        source data
    fpass: Sequence[int, int]
        catoff frequencies.
        fpass[0] is low catoff frequency.
        fpass[1] is high catoff frequency.
    fs: int
        sampling frequency

    Returns
    -------
    np.ndarray:
        filtered data
    """
    assert fpass[0] > 0, 'fpass[0] must be bigger than 0'
    yf = fft(y)
    yf /= (len(y)/2.)
    yf[0] = yf[0] / 2.

    freq = np.linspace(0, fs, len(y))
    yf[freq < fpass[0]] = 0
    yf[freq > fpass[1]] = 0

    yd = ifft(yf)
    yd = np.real(yd * len(y))

    return yd
