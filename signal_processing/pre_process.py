# ========== Packages ==========
import mne
from autoreject import (get_rejection_threshold, AutoReject)

# ========== Functions ==========
def filter_raw_data(raw,l_freq=0.5,h_freq=50,eog_remove=True):
    """
    Apply FIR bandpass filter and remove EOG noise.

    Parameters
    ----------
    raw: Raw-type (MNE-Python) EEG file
    l_freq (optional): A float or an integer for low cut-off frequency for the filter
    h_freq (optional): A float or an integer for high cut-off frequency for the filter
    eog_remove (optional): A boolean whether to remove EOG noise or not

    Returns
    -------
    filt: Raw-type (MNE-Python) EEG file
    """
    filt = raw.copy().load_data().filter(l_freq,h_freq).notch_filter([50])

    if eog_remove == True:
        eog_projs, _ = mne.preprocessing.compute_proj_eog(filt,n_grad=0,n_mag=0,n_eeg=1,reject=None,no_proj=True,ch_name=["EXG1","EXG2","EXG3","EXG4","EXG5","EXG6","EXG7","EXG8"])
        filt.add_proj(eog_projs,remove_existing=True)
        filt.apply_proj()
        filt.drop_channels(["EXG1","EXG2","EXG3","EXG4","EXG5","EXG6","EXG7","EXG8","Status"])

    return filt

def artefact_rejection(filt,subjectname,epo_duration=5):
    """
    Convert Raw file to Epochs and conduct artefact rejection/augmentation on the signal.

    Parameters
    ----------
    filt: Raw-type (MNE-Python) EEG file
    subjectname: A string for subject's name
    epo_duration (optional): An integer for the duration for epochs

    Returns
    -------
    epochs: Epochs-type (MNE-Python) EEG file
    """
    epochs = mne.make_fixed_length_epochs(filt, duration=epo_duration, preload=True)

    epochs.average().plot()
    epochs.plot_image(title="GFP without AR ({})".format(subjectname))

    reject_criteria = get_rejection_threshold(epochs)
    print('Dropping epochs with rejection threshold:',reject_criteria)
    epochs.drop_bad(reject=reject_criteria)

    ar = AutoReject(thresh_method='random_search',random_state=1)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    reject_log.plot('horizontal')

    epochs_ar.average().plot()
    epochs_ar.plot_image(title="GFP with AR ({})".format(subjectname))

    return epochs_ar