import numpy as np
from scipy import stats
import pandas as pd

def paired_Ttest(df_psd,conditions):
    bands = df_psd['Frequency band'].unique()
    regions = df_psd.drop(columns=['Subject','Frequency band','Condition']).columns.to_numpy()
    df_pvals = pd.DataFrame(index=regions, columns=bands)
    significant_locs = []

    for band in bands:
        df_psd_band = df_psd[df_psd['Frequency band'] == band]
        df_psd_band_cond1 = df_psd_band[df_psd_band['Condition'] == conditions[0]]\
                                .drop(columns=['Subject','Frequency band','Condition'])
        df_psd_band_cond2 = df_psd_band[df_psd_band['Condition'] == conditions[1]]\
                                .drop(columns=['Subject','Frequency band','Condition'])
        for region in df_psd_band_cond1.columns:
            _,df_pvals[band][region] = stats.ttest_rel(df_psd_band_cond1[region], df_psd_band_cond2[region])

        sign_idx = df_pvals.index[df_pvals[band]<=0.05].to_numpy()
        sign_pvals = df_pvals[df_pvals[band]<=0.05][band].to_numpy()
        if len(sign_idx) != 0:
            print('Significant changes of',band,'are:')
        for i in range(len(sign_idx)):
            print(sign_idx[i],'with p-value of',sign_pvals[i])
            significant_locs = np.append(significant_locs,{band: sign_idx[i]})

    return [df_pvals,significant_locs]