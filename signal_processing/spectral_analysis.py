# ========== Packages ==========
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# ========== Functions ==========
def calculate_psd(epochs, subjectname, fminmax=[1,50], method='welch', window='hamming',
                  window_duration=2, window_overlap=0.5, zero_padding=3, tminmax=[None, None],
                  verbose=False, plot=True):
    """
    Calculate power spectrum density with FFT/Welch's method and plot the result.

    Parameters
    ----------
    epochs: Epochs-type (MNE-Python) EEG file
    subjectname: A string for subject's name
    fminmax (optional): The minimum and maximum frequency range for estimating Welch's PSD
    window (optional): The window type for estimating Welch's PSD
    window_duration (optional): An integer for the length of that window
    window_overlap (optional): A float for the percentage of windows size 
                                for overlap between the windows
    zero-padding (optional): A float for coefficient times window size for zero-pads
    tminmax (optional): A list of first and last timepoint of the epoch to include;
                        uses all epoch by default

    Returns
    -------
    psds: An array for power spectrum density values
    freqs: An array for the corresponding frequencies
    """
    # Calculate window size in samples and window size x coefs for overlap and zero-pad
    window_size = int(epochs.info['sfreq']*window_duration)
    n_overlap = int(window_size*window_overlap)
    n_zeropad = int(window_size*zero_padding)

    # N of samples from signal equals to windows size
    n_per_seg = window_size

    # N of samples for FFT equals N of samples + zero-padding samples
    n_fft = n_per_seg + n_zeropad

    # Calculate PSD with Welch's method
    spectrum = epochs.compute_psd(method=method, fmin=fminmax[0], fmax=fminmax[1], 
                                  n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap,
                                  window=window, tmin=tminmax[0], tmax=tminmax[1],
                                  verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Unit conversion from V^2/Hz to uV^2/Hz
    psds = psds*1e12

    # If true, print all the parameters involved in PSD calculation
    if verbose == True:
        print("---\nPSD ({}) calculation\n".format(method))
        print(spectrum)
        print('Time period:', str(tminmax))
        print('Window type:', window)
        print('Window size:', window_size)
        print('Overlap:', n_overlap)
        print('Zero-padding:', n_zeropad)
        print('\nSamples per segment:', n_per_seg)
        print('Samples for FFT:', n_fft)
        print('Frequency resolution:', freqs[1]-freqs[0], 'Hz')

    # If true, plot average PSD for all epochs and channels with channel PSDs
    if plot == True:
        plt.figure(figsize=(5,3), dpi=100)
        plt.plot(freqs,np.transpose(psds.mean(axis=(0))),color='black',alpha=0.1)
        plt.plot(freqs,psds.mean(axis=(0, 1)),color='blue',alpha=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (uV\u00b2/Hz)')
        plt.title("PSD,{} ({})".format(method, subjectname))
        plt.xlim(fminmax)
        plt.ylim(0,None)
        plt.grid(linewidth=0.2)
        plt.show()

    return [psds,freqs]

def signal_quality_check(psds,freqs,band,b_name,subjectname,epochs):
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
        im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p1,epochs.info,axes=ax1,vlim=[vmin,vmax],show=False)
        im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p2,epochs.info,axes=ax2,vlim=[vmin,vmax],show=False)
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

def bandpower_per_channel(psds,freqs,band,b_name,subjectname,epochs,
                          ln_normalization=False,verbose=True):
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
    _ = signal_quality_check(psds,freqs,band,b_name,subjectname,epochs)
    
    # Average all epochs together for each channels' PSD values
    psds_per_ch = psds.mean(axis=(0))

    # Pick only PSD values which are within the frequency band of interest
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    psds_band_per_ch = psds_per_ch[:,idx_band]

    # Average the PSD values within the band together to get bandpower for each channel
    bp_per_ch = psds_band_per_ch.mean(axis=(1))

    # If true, normalise the BP with natural logarithm transform
    if ln_normalization == True:
        bp_per_ch = np.log(bp_per_ch)
    
    if verbose == True:
        print('Finding bandpower within {} Hz with Ln normalisation set to {}'.format(
               band, str(ln_normalization)))

    return bp_per_ch

def calculate_asymmetry_ch(df_psd_band,left_ch,right_ch):
    """
    Calculate asymmetry between brain hemispheres.

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values (for each region/channel) per subject for one band
    left_ch: A string for the left channel (or region)
    right_ch: A string for the right channel (or region)

    Returns
    -------
    df_asymmetry: A dataframe for calculated asymmetry for all the subjects
    """
    df_asymmetry = (df_psd_band[left_ch] - df_psd_band[right_ch])/(df_psd_band[left_ch] + df_psd_band[right_ch])*100

    return df_asymmetry

def find_ind_band(spectrum, freqs, freq_interest=[7, 14], bw_size=6):
    # Get indexes of band of interest
    freq_interest_idx = np.where(np.logical_and(freqs>=freq_interest[0],
                        freqs<=freq_interest[1]))
    
    # Find maximum amplitude (peak width) in that bandwidth
    pw = np.max(spectrum[freq_interest_idx])

    # Find center frequency index and value where the peak is
    cf_idx = np.where(spectrum == pw)
    cf = float(freqs[cf_idx])
    
    # Get bandwidth range for the band np.round(bw[0], 4)
    bw = [np.round(cf-bw_size/2, 4), np.round(cf+bw_size/2, 4)]

    # Find individual bandpower indexes based on the binsize
    bp_idx = np.logical_and(freqs>=bw[0], freqs<=bw[1])

    # Average the PSD values in these indexes together to get bandpower
    abs_bp = spectrum[bp_idx].mean()

    # Calculate relative bandpower
    rel_bp = abs_bp / spectrum.mean()

    return cf, pw, bw, abs_bp, rel_bp

def find_bp(spectrum, freqs, bw):
    # Find bandpower indexes based on the binsize
    bp_idx = np.logical_and(freqs>=bw[0], freqs<=bw[1])

    # Average the PSD values in these indexes together to get bandpower
    abs_bp = spectrum[bp_idx].mean()

    # Calculate relative bandpower
    rel_bp = abs_bp / spectrum.mean()

    return abs_bp, rel_bp