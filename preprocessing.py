from typing import Sequence, Tuple
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy import signal

__all__ = ['lpf', 'hpf', 'bpf']

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
    yf = fft(y.copy())
    freq = fftfreq(len(y), 1./fs)
    idx = np.logical_or(freq > fpass, freq < -fpass)
    yf[idx] = 0.

    yd = ifft(yf)
    yd = np.real(yd)

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
    yf = fft(y.copy())
    freq = fftfreq(len(y), 1./fs)
    idx = np.logical_and(freq < fpass, freq > -fpass)
    yf[idx] = 0.

    yd = ifft(yf)
    yd = np.real(yd)

    return yd



def bpf(y:np.ndarray, fpass:Sequence[Tuple[int, int]], fs:int) -> np.ndarray:
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
    yf = fft(y.copy())
    freq = fftfreq(len(y), 1./fs)
    idx_0 = np.logical_and(freq < fpass[0], freq > -fpass[0])
    idx_1 = np.logical_or(freq > fpass[1], freq < -fpass[1])
    idx = np.logical_or(idx_0, idx_1)
    yf[idx] = 0.

    yd = ifft(yf)
    yd = np.real(yd)

    return yd
