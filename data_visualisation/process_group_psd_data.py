# ========== Packages ==========
import mne, os
import pandas as pd
from basic.stats import apply_stat_test
from basic.arrange_data import *

# ========== Functions ==========
def read_group_psd_data(psd_reg_folder,psd_ch_folder,psd_faa_folder,exp_folder,non_responders=None):
    # Locate all PSD files (regions, channels and asymmetry) and save their information
    dir_inprogress_reg,b_names_reg,condition_reg = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
    dir_inprogress_ch,b_names_ch,condition_ch = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
    for i in range(len(exp_folder)):
        [dir_inprogress_reg[i],b_names_reg[i],condition_reg[i]] = read_excel_psd(exp_folder[i],psd_reg_folder,verbose=False)
        [dir_inprogress_ch[i],b_names_ch[i],condition_ch[i]] = read_excel_psd(exp_folder[i],psd_ch_folder,verbose=False)
    [dir_inprogress_faa,filenames_faa,condition_faa] = read_excel_psd('',psd_faa_folder,verbose=False)

    # Get one epochs file for later topographical plots' electrode placement information
    dir_inprogress_epo = os.path.join(r"Data\Clean",exp_folder[0])
    _, subject_names = read_files(dir_inprogress_epo,"_clean-epo.fif",verbose=False)
    epochs = mne.read_epochs(fname='{}\{}_clean-epo.fif'.format(dir_inprogress_epo,subject_names[0]),verbose=False)

    # Read all REGIONAL spectral data and save to dataframe
    df_psd_reg = pd.DataFrame()
    for i in range(len(b_names_reg[0])):
        for n_exps in range(len(b_names_reg)):
            globals()[b_names_reg[n_exps][i]] = pd.read_excel('{}\{}.xlsx'\
                                                .format(dir_inprogress_reg[n_exps],b_names_reg[n_exps][i]))\
            .assign(**{'Frequency band': condition_reg[n_exps][i][1],'Condition': condition_reg[n_exps][i][0]})
            df_psd_reg = pd.concat([df_psd_reg,globals()[b_names_reg[n_exps][i]]])

    # Read all CHANNELS spectral data and save to dataframe
    df_psd_ch = pd.DataFrame()
    for i in range(len(b_names_ch[0])):
        for n_exps in range(len(b_names_ch)):
            globals()[b_names_ch[n_exps][i]] = pd.read_excel('{}\{}.xlsx'
                                                            .format(dir_inprogress_ch[n_exps],b_names_ch[n_exps][i]))\
                .assign(**{'Frequency band': condition_ch[n_exps][i][1],'Condition': condition_ch[n_exps][i][0]})
            df_psd_ch = pd.concat([df_psd_ch,globals()[b_names_ch[n_exps][i]]])

    # (WIP) Read electrode pairs' data for frontal alpha asymmetry (FAA)
    df_faa = pd.DataFrame()
    for i in range(len(filenames_faa)):
        df_faa_temp = pd.read_excel('{}\{}.xlsx'.format(dir_inprogress_faa,filenames_faa[i]))\
            .assign(**{'Frequency band':'FAA','Condition': condition_faa[i][0].removesuffix('_frontal_asymmetry')})
        df_faa = pd.concat([df_faa,df_faa_temp])

    # Option to remove some participants from further analysis (ex. removing non-responders of treatment)
    if non_responders != None:
        df_psd_reg = df_psd_reg[df_psd_reg['Subject'].str.contains(non_responders) == False]
        df_psd_ch = df_psd_ch[df_psd_ch['Subject'].str.contains(non_responders) == False]
        df_faa = df_faa[df_faa['Subject'].str.contains(non_responders) == False]
    
    return [df_psd_reg,df_psd_ch,epochs]

def export_group_psd_comparison(psd_reg_folder,psd_ch_folder,df_psd_reg,df_psd_ch,stat_test,
                                condition_codes_comparisons,verbose=True):
    # Apply statistical test on the spectral data (regional and channels) and export results
    for condition in condition_codes_comparisons:
        df_reg_desc,df_reg_pvals,df_reg_statvals,_ = apply_stat_test(df_psd_reg,condition,stat_test=stat_test,verbose=verbose)
        df_ch_desc,df_ch_pvals,df_ch_statvals,_ = apply_stat_test(df_psd_ch,condition,stat_test=stat_test,verbose=verbose)
        try:
            os.makedirs(os.path.join(psd_reg_folder,''))
        except FileExistsError:
            pass
        with pd.ExcelWriter(r"{}\{}_results_{}-{}.xlsx".format(psd_reg_folder,stat_test,condition[0],condition[1])) as writer:
            df_reg_desc.to_excel(writer, sheet_name='Descripitives')
            df_reg_pvals.to_excel(writer, sheet_name='P-values')
            df_reg_statvals.to_excel(writer, sheet_name='Stat-values')
        try:
            os.makedirs(os.path.join(psd_ch_folder,''))
        except FileExistsError:
            pass
        with pd.ExcelWriter(r"{}\{}_results_{}-{}.xlsx".format(psd_ch_folder,stat_test,condition[0],condition[1])) as writer:
            df_ch_desc.to_excel(writer, sheet_name='Descripitives')
            df_ch_pvals.to_excel(writer, sheet_name='P-values')
            df_ch_statvals.to_excel(writer, sheet_name='Stat-values')

def add_clinical_data_to_psd(df_psd,clinical_data_path,rstrip='_EOC'):
    # Modify PSD dataframe and read in clinical data from Excel file
    df_psd['Subject'] = df_psd['Subject'].map(lambda x: x.rstrip(rstrip))
    df_psd = df_psd.set_index('Subject')
    clinicaldata = pd.read_excel(clinical_data_path).set_index('Subject')

    # Concatenate clinical data to EEG data
    df_psd_withclinical = pd.DataFrame()
    for band in df_psd['Frequency band'].unique():
        temp_df = pd.concat([df_psd[df_psd['Frequency band']==band],clinicaldata],axis=1)
        df_psd_withclinical = pd.concat([df_psd_withclinical,temp_df])

    return df_psd_withclinical

def apply_comparison_pairs_to_data(df_psd_withclinical,condition_codes_comparisons):
    # Create comparison dataframe for PSD and clinical data for each comparison pair for each band
    df_psd_withclinical_comparison = pd.DataFrame()
    for band in df_psd_withclinical['Frequency band'].unique():
        band_df = df_psd_withclinical[df_psd_withclinical['Frequency band']==band]
        comparison_df = pd.DataFrame()
        for comparison in condition_codes_comparisons:
            condition1_df = band_df[band_df['Condition']==comparison[0]]\
                            .reset_index(drop=True).drop(columns=['Frequency band','Condition'])
            condition2_df = band_df[band_df['Condition']==comparison[1]]\
                            .reset_index(drop=True).drop(columns=['Frequency band','Condition'])
            temp_comparison_df = condition2_df.subtract(condition1_df)
            temp_comparison_df['Condition'] = str(comparison)
            comparison_df = pd.concat([comparison_df,temp_comparison_df])
        comparison_df['Frequency band'] = band
        df_psd_withclinical_comparison = pd.concat([df_psd_withclinical_comparison, comparison_df])

    return df_psd_withclinical_comparison