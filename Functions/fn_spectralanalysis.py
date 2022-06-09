# ========== Packages ==========
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# ========== Functions ==========
def calculate_psd(epochs,subjectname,epo_duration=5):
    """
    Calculate power spectrum density with FFT/Welch's method and plot the result.

    Parameters
    ----------
    epochs: Epochs-type (MNE-Python) EEG file
    subjectname: A string for subject's name
    epo_duration (optional): An integer for the duration for epochs

    Returns
    -------
    psds: An array for power spectrum density values
    freqs: An array for corresponding frequencies
    """
    # Set time window and frequency range
    tmin = 0
    tmax = epo_duration
    fmin = 1
    fmax = 100
    sfreq = epochs.info['sfreq']

    # Calculate PSD with Welch's method
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        n_fft=int(sfreq * (tmax - tmin)),
        n_overlap=0, n_per_seg=None,
        tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax,
        window='boxcar',
        verbose=False)

    # Unit conversion from V^2/Hz to uV^2/Hz
    psds = psds*1e12

    # Plot average PSD for all epochs and channels (only for plot)
    psds_mean_all = psds.mean(axis=(0, 1))

    sns.set_style("darkgrid",{'font.family': ['sans-serif']})
    plt.figure()
    plt.plot(freqs,psds_mean_all)
    plt.fill_between(freqs,psds_mean_all)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (uV\u00b2/Hz)')
    plt.title("PSD ({})".format(subjectname))
    plt.xlim(1,40)
    plt.ylim(0,1.1*max(psds_mean_all))

    return [psds,freqs]

def siqnal_quality_check(psds,freqs,band,b_name,subjectname,epochs):
    """
    Plot topographically PSD values and calculate median absolute deviation
    for the first and second half of the signal for signal reliability control.

    Parameters
    ----------
    psds: An array for power spectrum density values
    freqs: An array for corresponding frequencies
    band: A list of lower and higher frequency for the frequency band in interest
    b_name: A string for frequency band in interest
    subjectname: A string for subject's name
    epochs: Epochs-type (MNE-Python) EEG file

    Returns
    -------
    psd_max_mad_error: A float for max median absolute deviation for PSD values
    """
    # Divide the epochs into two
    idx_mid_epoch = round(len(psds[:,0,0])/2)
    psds_p1 = psds[:idx_mid_epoch,:,:]
    psds_p2 = psds[idx_mid_epoch:,:,:]

    # Define the bandpower indices
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Get only the bandpower values that we need for both parts
    psd_band_ch_p1 = psds_p1[:,:,idx_band]
    psd_band_mean_epochs_p1 = psd_band_ch_p1.mean(axis=(2))
    psd_band_ch_p2 = psds_p2[:,:,idx_band]
    psd_band_mean_epochs_p2 = psd_band_ch_p2.mean(axis=(2))

    # Find Median Absolute Deviation along all the epochs for each channel
    psd_band_mean_epochs = np.concatenate((psd_band_mean_epochs_p1, psd_band_mean_epochs_p2))
    psd_median = np.median(psd_band_mean_epochs,axis=0)
    psd_mad = stats.median_abs_deviation(psd_band_mean_epochs,axis=0)

    # For each epoch in all channels, calculate the Z-score (MAD error) using median and MAD
    psd_mad_error_epochs_per_ch_p1 = np.zeros(shape=psd_band_mean_epochs_p1.shape)
    psd_mad_error_epochs_per_ch_p2 = np.zeros(shape=psd_band_mean_epochs_p2.shape)
    for i in range(len(psds[0,:,0])):
        psd_mad_error_epochs_per_ch_p1[:,i] = abs(psd_band_mean_epochs_p1[:,i] - psd_median[i])/psd_mad[i]
        psd_mad_error_epochs_per_ch_p2[:,i] = abs(psd_band_mean_epochs_p2[:,i] - psd_median[i])/psd_mad[i]

    # Average the Z-scores (MAD errors) for all epochs for the two parts -> get score for 19 channels
    psd_mad_error_per_ch_p1 = np.mean(psd_mad_error_epochs_per_ch_p1,axis=0)
    psd_mad_error_per_ch_p2 = np.mean(psd_mad_error_epochs_per_ch_p2,axis=0)

    # Find the average Z-score (MAD error) for the whole scalp (all channels together) for the two parts
    psd_mad_error_avg = [np.round(np.mean(psd_mad_error_per_ch_p1),3),
                         np.round(np.mean(psd_mad_error_per_ch_p2),3)]
    
    # Get the maximum of the two (for outputting it later)
    psd_max_mad_error = np.max(psd_mad_error_avg)

    # Find the average PSD values for each channel for both parts
    psds_all_channels_p1 = psds_p1.mean(axis=(0))
    psd_band_ch_p1 = psds_all_channels_p1[:,idx_band]
    psd_band_mean_ch_p1 = psd_band_ch_p1.mean(axis=(1))
    psds_all_channels_p2 = psds_p2.mean(axis=(0))
    psd_band_ch_p2 = psds_all_channels_p2[:,idx_band]
    psd_band_mean_ch_p2 = psd_band_ch_p2.mean(axis=(1))
    
    # Visually display the bandpower for both parts for visual inspection
    vmin = min([min(psd_band_mean_ch_p1),min(psd_band_mean_ch_p2)])
    vmax = max([max(psd_band_mean_ch_p1),max(psd_band_mean_ch_p2)])

    if psd_max_mad_error >= 2:
        sns.set_style("white",{'font.family': ['sans-serif']})
        fig,(ax1,ax2) = plt.subplots(ncols=2)
        fig.suptitle("MAD >= 2 ! Quality control for {} ({})".format(b_name,subjectname),y=1.1,x=0.575)
        im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p1,epochs.info,axes=ax1,vmin=vmin,vmax=vmax,show=False)
        im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p2,epochs.info,axes=ax2,vmin=vmin,vmax=vmax,show=False)
        ax1.set_title("Epochs 0-{}\nAvg MAD error = {}".format(idx_mid_epoch-1,psd_mad_error_avg[0]))
        ax2.set_title("Epochs {}-{}\nAvg MAD error = {}".format(idx_mid_epoch,len(psds[:,0,0]),psd_mad_error_avg[1]))
        ax_x_start = 0.95
        ax_x_width = 0.04
        ax_y_start = 0.1
        ax_y_height = 0.9
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_ylabel('uV\u00b2/Hz'); 

    return psd_max_mad_error

def bandpower_per_channel(psds,freqs,band,b_name,subjectname,epochs):
    """
    Find frequency band power in interest for all the channels.

    Parameters
    ----------
    psds: An array for power spectrum density values
    freqs: An array for corresponding frequencies
    band: A list of lower and higher frequency for the frequency band in interest
    b_name: A string for frequency band in interest
    subjectname: A string for subject's name
    epochs: Epochs-type (MNE-Python) EEG file

    Returns
    -------
    psd_band_mean_ch: An array for a frequency band power values for all the channels.
    """
    # Calculate the MAD error (z-score) of the bandpower to be sure of the quality
    psd_max_mad_error = siqnal_quality_check(psds,freqs,band,b_name,subjectname,epochs)
    
    low, high = band
    psds_all_channels = psds.mean(axis=(0))
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    psd_band_ch = psds_all_channels[:,idx_band]
    psd_band_mean_ch = psd_band_ch.mean(axis=(1))

    # # If the error is larger than 2, print it as a result (visual inspection)
    # if psd_max_mad_error < 2:
    #     print(subjectname,b_name,"MAD error is OK:",psd_max_mad_error)
    # else:
    #     print(subjectname,b_name,"MAD error is NOT OK:",psd_max_mad_error)

    return psd_band_mean_ch