
import numpy as np
import matplotlib.pyplot as plt


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
    n = len(y)
    spectrum = np.fft.fftshift(np.fft.fft(y) / n)
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
        fig, axs = plt.subplots(2, 1)
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
