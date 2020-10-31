
import numpy as np
from utils import get_random_range
from utils import uniform_rand


def _real_signal(phi, am=1):
    return np.cos(phi) * am


def _complex_signal(phi, am=1):
    return np.exp(1j * phi) * am


def _iq_signal(phi, am=1):
    return np.cos(phi) * am + np.sin(phi) * am


def _signal_gen(phi, am=1, numerical_type='complex'):
    # generate
    if numerical_type == 'complex':
        signal_gen = _complex_signal
    elif numerical_type == 'real':
        signal_gen = _real_signal
    elif numerical_type == 'iq':
        signal_gen = _iq_signal
    else:
        raise TypeError("Not a valid signal type that should be 'complex' or 'real'.")
    return signal_gen(phi, am)


def lfm(fs,
        bandwidth,
        timewidth,
        f0,
        sampling_time,
        init_phase=0,
        max_length=None,
        signal_type='complex',
        sweep_type='up'):
    #
    Ts = 1 / fs
    sample_len = int(sampling_time / Ts)
    k = bandwidth / timewidth if sweep_type == 'up' else - bandwidth / timewidth
    sub_sweep_num = int(sampling_time / timewidth)
    t = np.arange(0, timewidth, Ts)
    _N = len(t)

    # generate
    dtype = np.complex64 if signal_type == 'complex' else np.float32
    lfm_signal = np.zeros((sample_len,), dtype=dtype)
    i = -1
    for i in range(sub_sweep_num):
        phase = 2 * np.pi * f0 * t + np.pi * k * t ** 2 + init_phase
        lfm_signal[i * _N:(i + 1) * _N] = _signal_gen(phase, numerical_type=signal_type)
        init_phase = np.mod(phase[-1], 2 * np.pi)
    t = t[:sample_len - _N * (i + 1)]
    phase = 2 * np.pi * f0 * t + np.pi * k * t ** 2 + init_phase
    lfm_signal[(i + 1) * _N:] = _signal_gen(phase, numerical_type=signal_type)

    # cut
    if max_length:
        lfm_signal = lfm_signal[:max_length]

    return lfm_signal


def generate_random_lfm(fs, sampling_time, signal_type='complex', **kwargs):
    # get value range
    f0_range = get_random_range('f0_range', **kwargs)
    bw_range = get_random_range('bandwidth_range', **kwargs)
    # generate signal setting
    f0 = uniform_rand((1,), f0_range)[0]
    bw = uniform_rand((1,), bw_range)[0]
    init_phi = uniform_rand((1,), [0, 2*np.pi])[0]
    sig = lfm(fs, bandwidth=bw, timewidth=sampling_time, f0=f0,
              sampling_time=sampling_time, init_phase=init_phi,
              signal_type=signal_type)
    return sig, [f0, bw, init_phi]


def tone_signal(fs, f0, sampling_time, init_phase=0, signal_type='complex'):
    t = np.arange(0, sampling_time, 1/fs)
    phase = 2 * np.pi * f0 * t + init_phase
    tone_sig = _signal_gen(phase, numerical_type=signal_type)
    return tone_sig


def generate_random_tone(fs, sampling_time, signal_type='complex', **kwargs):
    # get value range
    f0_range = get_random_range('f0_range', **kwargs)
    # generate signal setting
    f0 = uniform_rand((1,), f0_range)[0]
    init_phi = uniform_rand((1,), [0, 2*np.pi])[0]
    sig = tone_signal(fs, f0, sampling_time, init_phi, signal_type)
    return sig, [f0, init_phi]


def multi_tone_signal(fs, f0s, sampling_time, init_phases=None, signal_type='complex'):
    assert isinstance(f0s, list)
    if init_phases is None:
        init_phases = [0] * len(f0s)
    multi_tone_sig = 0
    for f0, init_phase in zip(f0s, init_phases):
        multi_tone_sig += tone_signal(fs, f0, sampling_time, init_phase, signal_type)
    # normalize max value in real part and imaginary part to 1
    if signal_type == 'complex':
        multi_tone_sig = multi_tone_sig / np.max(np.abs(np.real(multi_tone_sig)))
    else:
        multi_tone_sig = multi_tone_sig / np.max(np.abs(multi_tone_sig))
    return multi_tone_sig


def generate_random_multi_tone(fs, sampling_time, signal_type='complex', **kwargs):
    # get value range
    f0_range = get_random_range('f0_range', **kwargs)
    tone_number = kwargs.get('tone_number', 2)
    # generate signal setting
    f0s = uniform_rand((tone_number,), f0_range)
    init_phis = uniform_rand((tone_number,), [0, 2*np.pi])
    sig = multi_tone_signal(fs, f0s, sampling_time, init_phis, signal_type)
    return sig, [f0s, init_phis]


def quantization(y, bit=16):
    assert isinstance(y, np.ndarray)
    if y.dtype == np.complex:
        s = np.real(y)
    else:
        s = y
    min_s = np.min(s)
    partition = (np.max(s) - min_s) / 2 ** bit
    if y.dtype == np.complex:
        quantized_y_real = np.round((np.real(y) - min_s) / partition) * partition + min_s
        quantized_y_imag = np.round((np.imag(y) - min_s) / partition) * partition + min_s
        return quantized_y_real + 1j * quantized_y_imag
    else:
        return np.round((y - min_s) / partition) * partition + min_s


def quantization_fixed(y, bit=16, ref_v=1):
    assert isinstance(y, np.ndarray)
    nbit = 2 ** bit
    if np.any(np.iscomplex(y)):
        return np.round(np.real(y) / ref_v * nbit) / nbit + \
               1j * np.round(np.imag(y) / ref_v * nbit) / nbit
    else:
        return np.round(y / ref_v * nbit) / nbit

