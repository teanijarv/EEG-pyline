from mne.utils import (logger, verbose, warn, _check_option, get_subjects_dir,
                     _mask_to_onsets_offsets, _pl, _on_missing, fill_doc)
from mne.filter import estimate_ringing_samples
from mne.viz.misc import _get_flim, _filter_ticks, adjust_axes, tight_layout, plt_show
import warnings
import mne
import numpy as np

_DEFAULT_ALIM = (-80, 10)

def plot_ideal_filter(freq, gain, axes=None, title='', flim=None, fscale='log',
                      alim=_DEFAULT_ALIM, color='r', alpha=0.5, linewidth=1, linestyle='--',
                      show=True):
    """Plot an ideal filter response.
    MNE-based function plot_ideal_filter() with changed linewidth parameter
    """
    import matplotlib.pyplot as plt
    my_freq, my_gain = list(), list()
    if freq[0] != 0:
        raise ValueError('freq should start with DC (zero) and end with '
                         'Nyquist, but got %s for DC' % (freq[0],))
    freq = np.array(freq)
    # deal with semilogx problems @ x=0
    _check_option('fscale', fscale, ['log', 'linear'])
    if fscale == 'log':
        freq[0] = 0.1 * freq[1] if flim is None else min(flim[0], freq[1])
    flim = _get_flim(flim, fscale, freq)
    transitions = list()
    for ii in range(len(freq)):
        if ii < len(freq) - 1 and gain[ii] != gain[ii + 1]:
            transitions += [[freq[ii], freq[ii + 1]]]
            my_freq += np.linspace(freq[ii], freq[ii + 1], 20,
                                   endpoint=False).tolist()
            my_gain += np.linspace(gain[ii], gain[ii + 1], 20,
                                   endpoint=False).tolist()
        else:
            my_freq.append(freq[ii])
            my_gain.append(gain[ii])
    my_gain = 10 * np.log10(np.maximum(my_gain, 10 ** (alim[0] / 10.)))
    if axes is None:
        axes = plt.subplots(1)[1]
    for transition in transitions:
        axes.axvspan(*transition, color=color, alpha=0.1)
    axes.plot(my_freq, my_gain, color=color, linestyle=linestyle, alpha=alpha,
              linewidth=linewidth, zorder=3)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    axes.set(ylim=alim, xlabel='Frequency (Hz)', ylabel='Amplitude (dB)',
             xscale=fscale)
    if xticks is not None:
        axes.set(xticks=xticks)
        axes.set(xticklabels=xticklabels)
    axes.set(xlim=flim)
    if title:
        axes.set(title=title)
    adjust_axes(axes)
    tight_layout()
    plt_show(show)
    return axes.figure



def plot_filter(h, sfreq, freq=None, gain=None, title=None, color='#1f77b4',
                flim=None, fscale='log', alim=_DEFAULT_ALIM, show=True,
                compensate=False, plot=('time', 'magnitude', 'delay'),
                axes=None, *, dlim=None, linewidth=0.9):
    """Plot properties of a filter.
    MNE-based function plot_filter() with added linewidth parameter and 
    """
    from scipy.signal import (
        freqz, group_delay, lfilter, filtfilt, sosfilt, sosfiltfilt)
    import matplotlib.pyplot as plt

    sfreq = float(sfreq)
    _check_option('fscale', fscale, ['log', 'linear'])
    if isinstance(plot, str):
        plot = [plot]
    for xi, x in enumerate(plot):
        _check_option('plot[%d]' % xi, x, ('magnitude', 'delay', 'time'))

    flim = _get_flim(flim, fscale, freq, sfreq)
    if fscale == 'log':
        omega = np.logspace(np.log10(flim[0]), np.log10(flim[1]), 1000)
    else:
        omega = np.linspace(flim[0], flim[1], 1000)
    xticks, xticklabels = _filter_ticks(flim, fscale)
    omega /= sfreq / (2 * np.pi)
    if isinstance(h, dict):  # IIR h.ndim == 2:  # second-order sections
        if 'sos' in h:
            H = np.ones(len(omega), np.complex128)
            gd = np.zeros(len(omega))
            for section in h['sos']:
                this_H = freqz(section[:3], section[3:], omega)[1]
                H *= this_H
                if compensate:
                    H *= this_H.conj()  # time reversal is freq conj
                else:
                    # Assume the forward-backward delay zeros out, which it
                    # mostly should
                    with warnings.catch_warnings(record=True):  # singular GD
                        warnings.simplefilter('ignore')
                        gd += group_delay((section[:3], section[3:]), omega)[1]
            n = estimate_ringing_samples(h['sos'])
            delta = np.zeros(n)
            delta[0] = 1
            if compensate:
                delta = np.pad(delta, [(n - 1, 0)], 'constant')
                func = sosfiltfilt
                gd += (len(delta) - 1) // 2
            else:
                func = sosfilt
            h = func(h['sos'], delta)
        else:
            H = freqz(h['b'], h['a'], omega)[1]
            if compensate:
                H *= H.conj()
            with warnings.catch_warnings(record=True):  # singular GD
                warnings.simplefilter('ignore')
                gd = group_delay((h['b'], h['a']), omega)[1]
                if compensate:
                    gd += group_delay((h['b'].conj(), h['a'].conj()), omega)[1]
            n = estimate_ringing_samples((h['b'], h['a']))
            delta = np.zeros(n)
            delta[0] = 1
            if compensate:
                delta = np.pad(delta, [(n - 1, 0)], 'constant')
                func = filtfilt
            else:
                func = lfilter
            h = func(h['b'], h['a'], delta)
        if title is None:
            title = 'SOS (IIR) filter'
        if compensate:
            title += ' (forward-backward)'
    else:
        H = freqz(h, worN=omega)[1]
        with warnings.catch_warnings(record=True):  # singular GD
            warnings.simplefilter('ignore')
            gd = group_delay((h, [1.]), omega)[1]
        title = 'FIR filter' if title is None else title
        if compensate:
            title += ' (delay-compensated)'

    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(plot), 1)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)
    if fig is None:
        fig = axes[0].get_figure()
    if len(axes) != len(plot):
        raise ValueError('Length of axes (%d) must be the same as number of '
                         'requested filter properties (%d)'
                         % (len(axes), len(plot)))

    t = np.arange(len(h))
    if dlim is None:
        dlim = np.abs(t).max() / 2.
        dlim = [-dlim, dlim]
    if compensate:
        n_shift = (len(h) - 1) // 2
        t -= n_shift
        assert t[0] == -t[-1]
        gd -= n_shift
    t = t / sfreq
    gd = gd / sfreq
    f = omega * sfreq / (2 * np.pi)
    sl = slice(0 if fscale == 'linear' else 1, None, None)
    mag = 10 * np.log10(np.maximum((H * H.conj()).real, 1e-20))

    if 'time' in plot:
        ax_time_idx = np.where([p == 'time' for p in plot])[0][0]
        axes[ax_time_idx].plot(t, h, color=color,linewidth=linewidth)
        axes[ax_time_idx].grid(visible=True,which='major',axis='both',linewidth = 0.15)
        axes[ax_time_idx].set(xlim=t[[0, -1]], xlabel='Time (s)',
                              ylabel='Amplitude', title=title)
    # Magnitude
    if 'magnitude' in plot:
        ax_mag_idx = np.where([p == 'magnitude' for p in plot])[0][0]
        axes[ax_mag_idx].plot(f[sl], mag[sl], color=color,
                              linewidth=linewidth, zorder=4)
        axes[ax_mag_idx].grid(visible=True,which='major',axis='both',linewidth = 0.15)
        if freq is not None and gain is not None:
            plot_ideal_filter(freq, gain, axes[ax_mag_idx], alpha=0.5, linestyle='--',linewidth=1.5,
                              fscale=fscale, flim=(0.001,sfreq/2), show=False)
        axes[ax_mag_idx].set(ylabel='Magnitude (dB)', xlabel='', xscale=fscale)
        if xticks is not None:
            axes[ax_mag_idx].set(xticks=xticks)
            axes[ax_mag_idx].set(xticklabels=xticklabels)
        axes[ax_mag_idx].set(xlim=flim, ylim=alim, xlabel='Frequency (Hz)',
                             ylabel='Amplitude (dB)')
    # Delay
    if 'delay' in plot:
        ax_delay_idx = np.where([p == 'delay' for p in plot])[0][0]
        axes[ax_delay_idx].plot(f[sl], gd[sl], color=color,
                                linewidth=linewidth, zorder=4)
        axes[ax_delay_idx].grid(visible=True,which='major',axis='both',linewidth = 0.15)
        # shade nulled regions
        for start, stop in zip(*_mask_to_onsets_offsets(mag <= -39.9)):
            axes[ax_delay_idx].axvspan(f[start], f[stop - 1],
                                       facecolor='k', alpha=0.05,
                                       zorder=5)
        axes[ax_delay_idx].set(xlim=flim, ylabel='Group delay (s)',
                               xlabel='Frequency (Hz)',
                               xscale=fscale)
        if xticks is not None:
            axes[ax_delay_idx].set(xticks=xticks)
            axes[ax_delay_idx].set(xticklabels=xticklabels)
        axes[ax_delay_idx].set(xlim=flim, ylim=dlim, xlabel='Frequency (Hz)',
                               ylabel='Delay (s)')

    adjust_axes(axes)
    tight_layout()
    plt_show(show)
    return fig


