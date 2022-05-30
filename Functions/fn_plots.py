import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statannotations.Annotator import Annotator

def plotPSDTopomaps(df_psd_ch,epochs,b_name,condition_names,fig_title):
    conditions = df_psd_ch['Condition'].unique()
    df_psd_ch_band = df_psd_ch[df_psd_ch['Frequency band'] == b_name].drop(columns=['Frequency band','Subject'])
    ar_psd_meanch_band_cond1 = df_psd_ch_band[df_psd_ch_band['Condition'] == conditions[0]]\
                                    .drop(columns=['Condition']).to_numpy().mean(axis=0)
    ar_psd_meanch_band_cond2 = df_psd_ch_band[df_psd_ch_band['Condition'] == conditions[1]]\
                                    .drop(columns=['Condition']).to_numpy().mean(axis=0)

    print('Plotting',conditions[0],'and',conditions[1],'for',b_name)

    vmin = min([min(ar_psd_meanch_band_cond1),min(ar_psd_meanch_band_cond2)])
    vmax = max([max(ar_psd_meanch_band_cond1),max(ar_psd_meanch_band_cond2)])
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
    fig.suptitle(fig_title,x=0.525)
    im,cm = mne.viz.plot_topomap(ar_psd_meanch_band_cond1,epochs.info,axes=ax1,vmin=vmin,vmax=vmax,show=False)
    im,cm = mne.viz.plot_topomap(ar_psd_meanch_band_cond2,epochs.info,axes=ax2,vmin=vmin,vmax=vmax,show=False)
    ax1.set_title(condition_names[0])
    ax2.set_title(condition_names[1])
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.125
    ax_y_height = 0.875
    
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_ylabel('uV\u00b2/Hz')

def plotBoxplotROI(df_psd,bands,region,condition_names,stat_test='t-test_paired',ast_loc='inside'):
    conditions = df_psd['Condition'].unique()
    x='Frequency band'
    hue='Condition'

    plt.figure()
    ax = sns.boxplot(x=x, y=region, hue=hue, order=bands, data=df_psd)

    pairs = [None]*len(bands)
    for i in range(len(bands)):
        pairs[i] = ((bands[i],conditions[0]),(bands[i],conditions[1]))

    annotator = Annotator(ax,pairs=pairs,data=df_psd, x=x, y=region,
                        hue=hue,plot="boxplot",order=bands)\
            .configure(test=stat_test,text_format='star',loc=ast_loc,text_offset=2)\
            .apply_and_annotate()

    ax.set(xlabel='Frequency band',ylabel='PSD (uV\u00b2/Hz)')
    ax.legend_.texts[0].set_text(condition_names[0])
    ax.legend_.texts[1].set_text(condition_names[1])

    if ast_loc == 'outside':
        ax.set_title('{} bandpowers boxplot'.format(region),y=1.125)
    else:
        ax.set_title('{} bandpowers boxplot'.format(region),y=1.025)

def plotBoxplotBand(df_psd,regions,band,condition_names,stat_test='t-test_paired',ast_loc='inside'):
    conditions = df_psd['Condition'].unique()

    df_psd_band = df_psd[df_psd['Frequency band'] == band]

    df_psd_band_final = pd.DataFrame()

    for region in regions:
        df_psd_temp = df_psd_band[[region,'Condition']].copy()
        df_psd_temp.rename(columns={region: band}, inplace=True)
        df_psd_temp['Region'] = region
        df_psd_band_final = pd.concat([df_psd_band_final,df_psd_temp]).reset_index(drop=True)

    plt.figure()
    ax = sns.boxplot(x='Region', y=band, hue='Condition', data=df_psd_band_final, order=regions)

    pairs = [None]*len(regions)
    for i in range(len(regions)):
        pairs[i] = ((regions[i],conditions[0]),(regions[i],conditions[1]))

    annotator = Annotator(ax,pairs=pairs,data=df_psd_band_final, x='Region', y=band,
                    hue='Condition',plot="boxplot",order=regions)\
            .configure(test=stat_test,text_format='star',loc=ast_loc,text_offset=2)\
            .apply_and_annotate()

    ax.set(xlabel='Regions',ylabel='PSD (uV\u00b2/Hz)')
    ax.legend_.texts[0].set_text(condition_names[0])
    ax.legend_.texts[1].set_text(condition_names[1])

    if ast_loc == 'outside':
        ax.set_title('{} power regional boxplot'.format(band),y=1.125)
    else:
        ax.set_title('{} power regional boxplot'.format(band),y=1.025)