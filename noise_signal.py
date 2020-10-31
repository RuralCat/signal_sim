
import numpy as np
from signal_gen import tone_signal
from utils import uniform_rand


def add_nonlinerity(y):
    return y + 0.001 * y**3


def add_white_noise(y, dbc=80):
    assert isinstance(y, np.ndarray)
    p = np.mean(np.abs(y)**2)
    snr = 10 ** ((dbc+10) / 20.0)
    n0 = p / snr
    noise = np.random.randn(len(y))
    if y.dtype == np.complex:
        noise = noise + 1j * np.random.randn(len(y))
        sigma = np.sqrt(n0 / 2)
    else:
        sigma = np.sqrt(n0)
    return y + sigma * noise


def add_dc_offset(y, dc_dbc=[50, 60], fs=1e9):
    min_dbc, max_dbc = dc_dbc
    init_phi = uniform_rand((1,), [0, 2*np.pi])
    t = np.arange(0, len(y)) / fs
    dc_f = fs / 1e4
    if np.any(np.iscomplex(y)):
        phase = 1j * (2 * np.pi * dc_f * t + init_phi)
    else:
        phase = 2 * np.pi * dc_f * t + init_phi
    dc_offset = np.exp(phase) * (max_dbc - min_dbc) / 2 + (min_dbc + max_dbc) / 2
    dc_offset = 10**(-dc_offset/20)

    return y + dc_offset


def add_image(y, image_dbc=[40, 50], fs=1e9):
    min_dbc, max_dbc = image_dbc
    init_phi = uniform_rand((1,), [0, 2*np.pi])
    st = len(y) / fs
    image_gain = tone_signal(fs, fs/1e4, st, init_phi, signal_type='real') * \
                 (max_dbc - min_dbc) / 2 + (max_dbc + min_dbc) / 2
    image_gain = 10**(-image_gain/20)
    return y + image_gain * np.conj(y)


def add_am_modulation(y, f0, fs):
    init_phi = uniform_rand((1,), [0, 2*np.pi])
    sampling_time = len(y) / fs
    am = tone_signal(fs, f0/1e4, sampling_time, init_phase=init_phi, signal_type='real')
    am = am * 0.2 + 0.8
    return am * y
