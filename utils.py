
import numpy as np
import matplotlib.pyplot as plt


def complex_to_array(y):
    assert isinstance(y, np.ndarray) and y.dtype == np.complex
    return np.stack([np.real(y), np.imag(y)], axis=-1)


def array_to_complex(y):
    assert isinstance(y, np.ndarray) and y.shape[-1] == 2
    return y[..., 0] + 1j * y[..., 1]


def uniform_rand(shape, a, b=None):
    """
    uniform random [a, b]
    """
    if isinstance(a, list) and len(a) == 2 and b is None:
        a, b = a[0], a[1]
    return np.random.rand(*shape) * (b - a) + a


def get_random_range(name, **kwargs):
    if name in kwargs:
        x_range = kwargs.get(name)
        if isinstance(x_range, list) and len(x_range) == 2:
            return x_range
        else:
            raise ValueError('value range should be passed by a list [a, b]')
    else:
        raise ValueError("Can not find value range '{}' in arguments.".format(name))


def get_spectrum(y, return_content='all'):
    """
    y: two types: (L,) or (..., L)
    return_content: 'all', 'sepc', 'am', 'phase'
    """
    n = y.shape[0] if y.ndim == 1 else y.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(y, axis=-1) / n, axes=-1)
    spectrum_am = np.abs(spectrum)
    spectrum_phase = np.angle(spectrum)
    if return_content == 'all':
        return spectrum, spectrum_am, spectrum_phase
    elif return_content == 'spec':
        return spectrum
    elif return_content == 'am':
        return spectrum_am
    elif return_content == 'phase':
        return spectrum_phase


def plot_spectrum(y, fs, normalize=False, show_phase=False, minimum_value=None):
    # y - real signal or complex signal
    n = len(y)
    spectrum, spectrum_am, spectrum_phase = get_spectrum(y)

    # plot
    # am to db
    spectrum_am_ = 20 * np.log10(spectrum_am + 10e-20)
    if normalize:
        spectrum_am_ = spectrum_am_ - np.max(spectrum_am_)
    xtick = (np.arange(0, fs, fs / n) - fs / 2) / 1e6

    if show_phase:
        # create figure
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        # set label
        axs[0].set_xlabel('Frequency(MHz)')
        axs[0].set_ylabel('Amplitude')
        axs[1].set_xlabel('Frequency(MHz)')
        axs[1].set_ylabel('Phase')
        #
        axs[0].plot(xtick, spectrum_am_)
        axs[0].grid()
        axs[1].plot(xtick, spectrum_phase)
        axs[1].grid()
    else:
        if minimum_value:
            spectrum_am_ = np.maximum(spectrum_am_, minimum_value)
        plt.plot(xtick, spectrum_am_)
        plt.xlabel('Frequency(MHz)')
        plt.ylabel('Amplitude')
        plt.grid()
    plt.show()


def compute_sfdr(y, img_y, show_infos=True):
    assert isinstance(y, np.ndarray) and isinstance(img_y, np.ndarray)
    if y.ndim > 2 and y.shape[-1] == 2:
        y = array_to_complex(y)
        img_y = array_to_complex(img_y)
    y_spec_am = get_spectrum(y, return_content='am')
    img_spec_am = get_spectrum(img_y, return_content='am')
    img_spec_am = 20 * np.log10(np.abs(y_spec_am - img_spec_am) + 1e-16)
    y_spec_am = 20 * np.log10(y_spec_am + 1e-16)

    sfdrs = np.max(y_spec_am, axis=-1) - np.max(img_spec_am, axis=-1)
    
    if show_infos:
        print("SFDR range (dbc): {:.2f} ~ {:.2f}, mean value: {:.2f}".format(
            np.min(sfdrs), np.max(sfdrs), np.mean(sfdrs)))
    return sfdrs


def compute_phase_error(y, img_y, data_type='tone', show_infos=True):
    assert isinstance(y, np.ndarray) and isinstance(img_y, np.ndarray)
    if y.ndim > 2 and y.shape[-1] == 2:
        y = array_to_complex(y)
        img_y = array_to_complex(img_y)
    if data_type == 'tone':
        _, ideal_am, ideal_phase = get_spectrum(y, return_content='all')
        _, img_am, img_phase = get_spectrum(img_y, return_content='all')
    elif data_type == 'lfm':
        pass
    

def plot_sfdr_freqs(sfdrs, data_type='lfm', fs=500e6, f_step=1e6, nb_sample_per_step=100, fixed_bw=10e6):
    # parse args
    naq_f = fs / 2
    if data_type == 'lfm':
        f0_range = np.arange(-naq_f+fixed_bw/2, naq_f-fixed_bw/2, f_step)
    else:
        f0_range = np.arange(-naq_f+f_step, naq_f+f_step, f_step)
    sfdrs = np.mean(np.reshape(sfdrs, (len(f0_range), nb_sample_per_step)), axis=-1)
    # plot
    plt.plot(f0_range/1e6, sfdrs)
    plt.xlabel('Frequency (MHz)')
    plt.grid()
    plt.show()

