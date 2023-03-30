# ========== Packages ==========
import os
import pandas as pd

# ========== Functions ==========
def read_files(dir_inprogress,filetype,exclude_subjects=[],verbose=True):
    """
    Get all the (EEG) file directories and subject names.

    Parameters
    ----------
    dir_inprogress: A string with directory to look for files
    filetype: A string with the ending of the files we are looking for (e.g. '.bdf')

    Returns
    -------
    file_dirs: A list of strings with file directories for all the (EEG) files
    subject_names: A list of strings with all the corresponding subject names
    """
    file_dirs = []
    subject_names = []
    for file in os.listdir(dir_inprogress):
        if file.endswith(filetype):
            file_dirs.append(os.path.join(dir_inprogress, file))
            subject_names.append(os.path.join(file).removesuffix(filetype))

    try:
        for excl_sub in exclude_subjects:
            for i in range(len(subject_names)):
                if excl_sub in subject_names[i]:
                    if verbose == True:
                        print('EXCLUDED SUBJECT: ',excl_sub,'in',subject_names[i],'at',file_dirs[i])
                    del subject_names[i]
                    del file_dirs[i]
                    break
    except:
        pass
    
    file_dirs = sorted(file_dirs)
    subject_names = sorted(subject_names)

    if verbose == True:
        print("Files in {} read in: {}".format(dir_inprogress,len(file_dirs)))

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

def df_channels_to_regions(df_psd_band, brain_regions):
    """
    Average channels together based on the defined brain regions.

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values for each channel per subject
    brain_regions: A dictionary of brain regions and EEG channels which they contain
    drop_cols: List of columns which are not channel PSD data

    Returns
    -------
    df_psd_reg_band: A dataframe with PSD values for each brain region per subject
    """

    df_psd_reg_band = pd.DataFrame()
    for region in brain_regions:
        df_temp = df_psd_band[brain_regions[region]].copy().mean(axis=1)
        df_psd_reg_band = pd.concat([df_psd_reg_band, df_temp], axis=1)
        
    df_psd_reg_band.columns = brain_regions.keys()
    df_psd_reg_band.index.name = 'Subject'

    return df_psd_reg_band

def read_excel_psd(exp_folder, psd_folder, condition_strsplit='_psd_', verbose=True):
    """
    Get all PSD file directories and corresponding bands and experiment conditions.

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
        condition[i] = b_names[i].split(condition_strsplit, 1)
    
    return [dir_inprogress,b_names,condition]

def create_results_folders(exp_folder, results_folder='Results', abs_psd=False,
                           rel_psd=False, fooof=False, erps=False):
    """
    Dummy way to try to pre-create folders for PSD results before exporting them

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    """
    if abs_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/Absolute PSD/channels'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/Absolute PSD/regions'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if rel_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/Relative PSD/channels'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/Relative PSD/regions'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if fooof == True:
        try:
            os.makedirs(os.path.join('{}/{}/FOOOF'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if erps == True:
        try:
            os.makedirs(os.path.join('{}/{}/ERP analysis'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    try:
        os.makedirs(os.path.join('{}/{}'.format(results_folder, exp_folder)))
    except FileExistsError:
        pass

def export_psd_results(df_psd_band,df_rel_psd_band,exp_folder,exp_condition,band,brain_regions,results_folder='Results'):
    """
    Export PSD results (for channels & regions) as Excel files

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values for each channel per subject
    df_rel_psd_band: A dataframe with relative PSD values for each channel per subject
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    exp_condition: A string for experiment short code (e.g. 'EC_00')
    band: The frequency band of the PSD values (e.g. 'Alpha')
    brain_regions: A dictionary of brain regions and EEG channels which they contain

    """
    # Save the PSD values for each channel for each band in Excel format
    df_psd_band.to_excel(r"{}/Absolute PSD/channels/{}/{}_psd_{}.xlsx".format(results_folder,exp_folder,exp_condition,band))
    df_rel_psd_band.to_excel(r"{}/Relative PSD/channels/{}/{}_rel_psd_{}.xlsx".format(results_folder,exp_folder,exp_condition,band))

    # Find regional band powers and save them to Excel as well
    df_psd_band_reg = df_channels_to_regions(df_psd_band,brain_regions)
    df_psd_band_reg.to_excel(r"{}/Absolute PSD/regions/{}/{}_psd_{}.xlsx".format(results_folder,exp_folder,exp_condition,band))
    df_rel_psd_band_reg = df_channels_to_regions(df_rel_psd_band,brain_regions)
    df_rel_psd_band_reg.to_excel(r"{}/Relative PSD/regions/{}/{}_rel_psd_{}.xlsx".format(results_folder,exp_folder,exp_condition,band))
