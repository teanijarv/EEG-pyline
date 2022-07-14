# ========== Packages ==========
import os
import pandas as pd

# ========== Functions ==========
def read_files(dir_inprogress,filetype,verbose=True):
    """
    Get all the EEG file directories and subject names.

    Parameters
    ----------
    dir_inprogress: A string with directory to look for files
    filetype: A string with the ending of the files we are looking for

    Returns
    -------
    file_dirs: A list of strings with file directories for all the EEG files
    subject_names: A list of strings with all the corresponding subject names
    """
    file_dirs = []
    subject_names = []
    for file in os.listdir(dir_inprogress):
        if file.endswith(filetype):
            file_dirs.append(os.path.join(dir_inprogress, file))
            subject_names.append(os.path.join(file).removesuffix(filetype))
    if verbose == True:
        print("Files in folder:",len(file_dirs))

    return [file_dirs, subject_names]

def array_to_df(subjectname,epochs,array_channels):
    """
    Convert channel-based array to Pandas dataframe with channels' and subjects' names.

    Parameters
    ----------
    subjectname: A string for subject's name
    epochs: Epochs-type (MNE-Python) EEG file
    array_channels: An array with values for each channel

    Returns
    -------
    df_channels: A dataframe with values for each channel
    """
    df_channels = pd.DataFrame(array_channels).T
    df_channels.columns = epochs.info.ch_names
    df_channels['Subject'] = subjectname
    df_channels.set_index('Subject', inplace=True)

    return df_channels

def df_channels_to_regions(df_channels):
    """
    Convert channel-based dataframe to region-based dataframe.

    Parameters
    ----------
    df_channels: A dataframe with values for each channel

    Returns
    -------
    df_regional: A dataframe with values for brain regions
    """
    df_frontal = df_channels[['Fp1','Fp2','AF3','AF4','F3','F4','F7','F8','Fz']].copy().mean(axis=1)
    df_temporal = df_channels[['FC5','FC6','T7','T8','CP5','CP6','P7','P8']].copy().mean(axis=1)
    df_centroparietal = df_channels[['FC1','FC2','C3','C4','Cz','CP1','CP2','P3','P4','Pz']].copy().mean(axis=1)
    df_occipital = df_channels[['PO3','PO4','O1','O2','Oz']].copy().mean(axis=1)
    df_regional = pd.concat([df_frontal,df_temporal,
                             df_centroparietal,df_occipital], axis=1)
    df_regional.columns = ['Frontal','Temporal','Centro-parietal','Occipital']

    # df_left_frontal = df_channels[['Fp1','AF3','F3','F7']].copy().mean(axis=1)
    # df_right_frontal = df_channels[['Fp2','AF4','F4','F8']].copy().mean(axis=1)
    # df_central = df_channels[['Fz','FC1','FC2','C3','C4','Cz']].copy().mean(axis=1)
    # df_parietal = df_channels[['CP1','CP2','P3','P4','Pz']].copy().mean(axis=1)
    # df_left_temporal = df_channels[['FC5','T7','CP5','P7']].copy().mean(axis=1)
    # df_right_temporal = df_channels[['FC6','T8','CP6','P8']].copy().mean(axis=1)
    # df_occipital = df_channels[['PO3','PO4','O1','O2','Oz']].copy().mean(axis=1)

    # df_regional = pd.concat([df_left_frontal,df_right_frontal,df_central,
    #                          df_parietal,df_left_temporal,df_right_temporal,
    #                          df_occipital], axis=1)
    # df_regional.columns = ['Left-frontal','Right-frontal','Central','Parietal',
    #                        'Left-temporal','Right-temporal','Occipital']

    return df_regional

def read_excel_psd(exp_folder,psd_folder,verbose=True):
    """
    Get all power spectrum density file directories and corresponding bands and experiment conditions.

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    psd_folder: A string with a relative directory to the results folder (e.g. 'Results\PSD\regions')

    Returns
    -------
    dir_inprogress: A string with directory to look for files
    b_names: A list of strings for frequency bands of the files
    condition: A list of strings for experiment conditions of the files
    """
    dir_inprogress = os.path.join(psd_folder,exp_folder)
    _, b_names = read_files(dir_inprogress,".xlsx",verbose=verbose)

    condition = [None]*len(b_names)
    for i in range(len(b_names)):
        condition[i] = b_names[i].split("_psd_", 1)
    
    return [dir_inprogress,b_names,condition]

def create_results_folders(exp_folder):
    # Dummy way to try to pre-create folders for PSD results
    try:
        os.makedirs(os.path.join(r"Results\Absolute PSD\channels",exp_folder))
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(r"Results\Absolute PSD\regions",exp_folder))
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(r"Results\Relative PSD\channels",exp_folder))
    except FileExistsError:
        pass
    try:
        os.makedirs(os.path.join(r"Results\Relative PSD\regions",exp_folder))
    except FileExistsError:
        pass

def export_psd_results(df_psd_band,df_rel_psd_band,exp_folder,exp_condition,band):
    # Save the PSD values for each channel for each band in Excel format
    df_psd_band.to_excel(r"Results\Absolute PSD\channels\{}\{}_psd_{}.xlsx".format(exp_folder,exp_condition,band))
    df_rel_psd_band.to_excel(r"Results\Relative PSD\channels\{}\{}_rel_psd_{}.xlsx".format(exp_folder,exp_condition,band))

    # Find regional band powers and save them to Excel as well
    df_psd_band_reg = df_channels_to_regions(df_psd_band)
    df_psd_band_reg.to_excel(r"Results\Absolute PSD\regions\{}\{}_psd_{}.xlsx".format(exp_folder,exp_condition,band))
    df_rel_psd_band_reg = df_channels_to_regions(df_rel_psd_band)
    df_rel_psd_band_reg.to_excel(r"Results\Relative PSD\regions\{}\{}_rel_psd_{}.xlsx".format(exp_folder,exp_condition,band))
