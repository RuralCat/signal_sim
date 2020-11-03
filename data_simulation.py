"""
    Data simulation for zero-IF receiver
"""

from utils import plot_spectrum, get_spectrum, compute_sfdr
from utils import uniform_rand
from signal_gen import generate_random_lfm
from signal_gen import generate_random_tone
from signal_gen import generate_random_multi_tone
from signal_gen import quantization_fixed
from signal_gen import lfm, tone_signal, multi_tone_signal
from noise_signal import *
import numpy as np
from tqdm import tqdm


def _check_and_set_default(name, default, **kwargs):
    v = kwargs.get(name, None)
    if v is None:
        kwargs.setdefault(name, default=default)


def generate_dataset(data_type='lfm',
                     nb_samples=1,
                     fs=1e9,
                     sample_length=10000,
                     numerical_type='complex',
                     return_am_phase=False,
                     badd_nonliner=True,
                     badd_am_modulation=True,
                     badd_image=True,
                     image_dbc_range=None,
                     badd_white_noise=True,
                     noise_dbc_range=None,
                     badd_dc_offset=True,
                     dc_dbc_range=None,
                     return_real_img=False,
                     verbose=0,
                     **kwargs):
    """
    data_type: 'lfm', 'tone', 'multi_tone'
    setting in kwargs:
        'lfm': 'f0_range', 'bandwidth_range',
        'tone': 'f0_range',
        'multi_tone': 'f0_range', 'tone_number'
        
    """
    # parse data setting
    # signal generation signal
    if data_type == 'lfm':
        _check_and_set_default('f0_range', [40e6, 80e6], **kwargs)
        _check_and_set_default('bandwidth_range', [10e6, 20e6], **kwargs)
        signal_gen = generate_random_lfm
    elif data_type == 'tone':
        _check_and_set_default('f0_range', [10e6, 100e6], **kwargs)
        signal_gen = generate_random_tone
    elif data_type == 'multi_tone':
        _check_and_set_default('f0_range', [10e6, 100e6], **kwargs)
        _check_and_set_default('tone_number', 3, **kwargs)
        signal_gen = generate_random_multi_tone
    # noise range
    if image_dbc_range is None:
        image_dbc_range = [40, 50]
    if noise_dbc_range is None:
        noise_dbc_range = [80, 90]
    if dc_dbc_range is None:
        dc_dbc_range = [50, 60]
    # sampling time
    sampling_time = sample_length / fs
        
    #
    dtype = np.complex64 if numerical_type == 'complex' else np.float32
    ideal_signals = np.zeros((nb_samples, sample_length), dtype=dtype)
    noise_signals = np.zeros((nb_samples, sample_length), dtype=dtype)
    settings = []
    irange = range(nb_samples)
    if verbose:
        irange = tqdm(irange, desc='dataset generating...')
    for i in irange:
        # generate signal
        sig, setting = signal_gen(fs, sampling_time, numerical_type, **kwargs)
        
        # amplitude modulation
        if badd_am_modulation:
            f0 = setting[0]
            sig = add_am_modulation(sig, f0, fs)
        
        # add non-linearity
        if badd_nonliner:
            sig = add_nonlinerity(sig)
        # add white noise
        if badd_white_noise:
            wgn_dbc = uniform_rand((1,), noise_dbc_range)[0]
            sig = add_white_noise(sig, dbc=wgn_dbc)
            setting.append(wgn_dbc)

        # add image & dc
        noise_sig = sig
        if badd_image:
            noise_sig = add_image(noise_sig, image_dbc_range, fs)
        if badd_dc_offset:
            noise_sig = add_dc_offset(noise_sig, dc_dbc_range, fs)
        ideal_signals[i] = sig
        noise_signals[i] = noise_sig
        settings.append(setting)
    
    if numerical_type == 'complex' and return_am_phase:
        ideal_signals = np.stack([np.abs(ideal_signals),
                                  np.angle(ideal_signals)], axis=-1)
        noise_signals = np.stack([np.abs(noise_signals),
                                  np.angle(noise_signals)], axis=-1)
    elif numerical_type == 'complex' and return_real_img:
        ideal_signals = np.stack([np.real(ideal_signals),
                                  np.imag(ideal_signals)], axis=-1)
        noise_signals = np.stack([np.real(noise_signals),
                                  np.imag(noise_signals)], axis=-1)
    
    return ideal_signals, noise_signals, settings


def generate_training_dataset(data_type='lfm',
                              fs=500e6,
                              f_step=1e6,
                              nb_sample_per_step=200,
                              sample_length=10000,
                              fixed_bw=10e6):
    # parse args
    naq_f = fs / 2
    if data_type == 'lfm':
        f0_range = np.arange(-naq_f+f_step/2, naq_f-fixed_bw, f_step)
    else:
        f0_range = np.arange(-naq_f+f_step/2, naq_f, f_step)
    nb_samples = len(f0_range) * nb_sample_per_step
    # simulate signals
    ideal_signals = np.zeros((nb_samples, sample_length, 2),
                             dtype=np.float32)
    noise_signals = np.zeros((nb_samples, sample_length, 2),
                             dtype=np.float32)
    settings = []
    for i, f0 in tqdm(enumerate(f0_range),
                      desc='dataset generating...', total=len(f0_range)):
        _ideal_signals, _noise_signals, _setttings = generate_dataset(
            data_type=data_type,
            nb_samples=nb_sample_per_step,
            fs=fs,
            sample_length=sample_length,
            numerical_type='complex',
            return_real_img=True,
            f0_range=[f0, f0+f_step],
            bandwidth_range=[fixed_bw, fixed_bw]
        )
        ideal_signals[i*nb_sample_per_step:(i+1)*nb_sample_per_step] = _ideal_signals
        noise_signals[i*nb_sample_per_step:(i+1)*nb_sample_per_step] = _noise_signals
        settings.extend(_setttings)
    inputs = np.reshape(ideal_signals, (nb_samples, sample_length//100, 100, 2))
    targets = np.reshape(ideal_signals-noise_signals, (nb_samples, sample_length//100, 200))
    return ideal_signals, noise_signals, inputs, targets, settings


def sim_v0():
    # setting: name(unit)
    # sampling rate:fs(GHz); band width: bw(MHz); time width: tw(us)
    # start frequency:f0(MHz); sampling time: st(us);
    fs = 1.0 * 1e9
    bw = 50.0 * 1e6
    tw = 1e-5
    f0 = 150 * 1e6
    st = 10e-6
    stype = 'lfm'
    if stype == 'lfm':
        y = lfm(fs, bw, tw, f0, st,
                init_phase=0,
                max_length=None,
                signal_type='complex')
    elif stype == 'tone':
        y = tone_signal(fs, f0, st, signal_type='complex')
    elif stype == 'multi_tone':
        y = multi_tone_signal(fs, [f0, -30e6], st, signal_type='complex')
    plot_spectrum(y, fs, normalize=False)
    print('signal power:', np.mean(np.abs(y) ** 2))
    freq_am = get_spectrum(y, return_content='am')
    max_freq_am = np.max(20 * np.log10(freq_am))
    
    # am modulation
    am = tone_signal(fs, f0 / 1e4, sampling_time=st, signal_type='real')
    am = am * 0.2 + 0.8
    y = am * y
    plot_spectrum(y, fs, normalize=False)

    # add non-linearity
    non_liner_y = add_nonlinerity(y)
    plot_spectrum(non_liner_y, fs, normalize=False)

    # white noise
    noise_y = add_white_noise(non_liner_y, dbc=80 + 10)
    plot_spectrum(noise_y, fs, normalize=False)

    # add dc offset
    dc_y = add_dc_offset(noise_y, dc_dbc=[40, 60], fs=fs)
    plot_spectrum(dc_y, fs, normalize=False)

    # add image
    img_y = add_image(dc_y, fs=fs)
    plot_spectrum(img_y, fs, normalize=False)
    
    # quantization
    quant_y = quantization_fixed(dc_y, bit=8)
    plot_spectrum(quant_y, fs, normalize=False)
    
    print('SFDR: ', compute_sfdr(noise_y, img_y))


def sim_v1():
    ideal_signals, noise_signals, _ = generate_dataset(
        data_type='lfm',
        nb_samples=10,
        fs=1e9,
        sample_length=10000,
        numerical_type='complex',
        return_am_phase=False,
        f0_range=[40e6, 80e6],
        bandwidth_range=[10e6, 20e6]
    )
    plot_spectrum(ideal_signals[0], fs=1e9, normalize=False)
    plot_spectrum(noise_signals[0], fs=1e9, normalize=False)
    

def sim_v2():
    ideal_signals, image_signals, inputs, targets, settings = generate_training_dataset(
        data_type='tone',
        nb_sample_per_step=1
    )
    sfdrs = compute_sfdr(ideal_signals, image_signals)


if __name__ == '__main__':
    sim_v2()
