# ========== Packages ==========
import numpy as np
from scipy import stats
import pandas as pd

# ========== Functions ==========
def apply_stat_test(df_psd,conditions,stat_test,verbose=False):
    """
    Perform a paired t-test on dataframe with two conditions.

    Parameters
    ----------
    df_psd: A Pandas dataframe of the power spectra values (for all the channels or regions)
    condition_comp_list: A list of two strings for experiment conditions codes to compare (e.g. ['EC_00','EC_06'])
    stat_test: The statistical test to be applied (available: 't-test_paired','Wilcoxon')

    Returns
    -------
    df_desc: A Pandas dataframe including descriptive measures for the compared data
    df_pvals: A Pandas dataframe of p-values from the performed test
    df_statistic: A Pandas dataframe of the test's statistic values
    significant_locs: An array of locations (regions/channels) where p <= 0.05
    """
    bands = df_psd['Frequency band'].unique()
    regions = df_psd.drop(columns=['Subject','Frequency band','Condition']).columns.to_numpy()
    df_desc = pd.DataFrame()
    df_pvals = pd.DataFrame(index=regions, columns=bands)
    df_statistic = pd.DataFrame(index=regions, columns=bands)
    significant_locs = []

    for band in bands:
        df_psd_band = df_psd[df_psd['Frequency band'] == band]
        df_psd_band_cond1 = df_psd_band[df_psd_band['Condition'] == conditions[0]]\
                                .drop(columns=['Subject','Frequency band','Condition'])
        df_psd_band_cond2 = df_psd_band[df_psd_band['Condition'] == conditions[1]]\
                                .drop(columns=['Subject','Frequency band','Condition'])

        df_cond1_desc = df_psd_band_cond1.describe().add_suffix('_'+band+'_'+conditions[0])
        df_cond2_desc = df_psd_band_cond2.describe().add_suffix('_'+band+'_'+conditions[1])
        df_desc = pd.concat([df_desc, pd.concat([df_cond1_desc, df_cond2_desc], axis=1)], axis=1)

        for region in df_psd_band_cond1.columns:
            if stat_test=='t-test_paired':
                df_statistic[band][region],df_pvals[band][region] = stats.ttest_rel(df_psd_band_cond1[region], df_psd_band_cond2[region])
            elif stat_test=='Wilcoxon':
                df_statistic[band][region],df_pvals[band][region] = stats.wilcoxon(df_psd_band_cond1[region], df_psd_band_cond2[region])
            elif stat_test=='t-test_ind':
                df_statistic[band][region],df_pvals[band][region] = stats.ttest_ind(df_psd_band_cond1[region], df_psd_band_cond2[region])
            else:
                print('No valid statistical test chosen')
        
        sign_idx = df_pvals.index[df_pvals[band]<=0.05].to_numpy()
        #sign_pvals = df_pvals[df_pvals[band][1]<=0.05][band].to_numpy()
        if ((len(sign_idx) != 0) & (verbose == True)):
            print(stat_test+':',conditions,band,'significant at',sign_idx)
        for i in range(len(sign_idx)):
            significant_locs = np.append(significant_locs,{band: sign_idx[i]})

    return [df_desc,df_pvals,df_statistic,significant_locs]