import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal

def lpf(y, fpass, fs):
    yf = fft(y)
    yf /= (len(y)/2.)
    yf[0] = yf[0] / 2.

    freq = np.linspace(0, fs, len(y))
    yf[freq > fpass] = 0

    yd = ifft(yf)
    yd = np.real(yd * len(y))

    return yd

def hpf(y, fpass, fs):
    yf = fft(y)
    yf /= (len(y)/2.)
    yf[0] = yf[0] / 2.

    freq = np.linspace(0, fs, len(y))
    yf[freq < fpass] = 0

    yd = ifft(yf)
    yd = np.real(yd * len(y))

    return yd



def bpf(y, fpass, fs):
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

