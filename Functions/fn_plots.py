import mne, os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statannotations.Annotator import Annotator
from Functions.fn_stats import *

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

def plot_boxplot_location(df_psd,bands,region,condition_comp_list,condition_legend,fnt=['sans-serif',8,10],stat_test='t-test_paired',ast_loc='inside',export=False):
    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    x='Frequency band'
    hue='Condition'

    plt.figure(dpi=100)
    ax = sns.boxplot(x=x, y=region, hue=hue, order=bands, data=df_psd,
                     flierprops=dict(markerfacecolor = '0.5', markersize = 3))
    
    pairs = []
    if stat_test=='t-test_paired':
        for i in range(len(condition_comp_list)):
            _,significant = paired_Ttest(df_psd[['Subject','Frequency band','Condition',region]],condition_comp_list[i])
            for j in range(len(significant)):
                sign_temp = list(significant[j].keys())[0]
                for s in range(len(bands)):
                    if sign_temp == bands[s]:
                        pairs.append(((*significant[j].keys(),condition_comp_list[i][0]),(*significant[j].keys(),condition_comp_list[i][1])))
    else:
        print(stat_test,'as a parameter for the statistical test is not supported.')
    
    if len(pairs) != 0:
        annotator = Annotator(ax,pairs=pairs,data=df_psd, x=x, y=region,
                            hue=hue,plot="boxplot",order=bands)\
                .configure(test=stat_test,loc=ast_loc,text_offset=2,text_format='star')\
                .apply_and_annotate()
    
    plt.legend(title='Condition', title_fontsize=fnt[2],fontsize=fnt[1])
    for i in range(len(condition_legend)):
        ax.legend_.texts[i].set_text(condition_legend[i])

    if ast_loc == 'outside':
        ax.set_title('{} bandpowers boxplot'.format(region),y=1.125,fontsize=fnt[2])
    else:
        ax.set_title('{} bandpowers boxplot'.format(region),y=1.025,fontsize=fnt[2])

    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel('PSD (µV\u00b2/Hz)', fontsize=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}.tiff'.format(region,bands,stat_test),dpi=300,bbox_inches='tight')
    
def plot_boxplot_band(df_psd,regions,band,condition_comp_list,condition_legend,fnt=['sans-serif',8,10],stat_test='t-test_paired',ast_loc='inside',export=False):
    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    x = 'Region'
    hue = 'Condition'
    
    df_psd_band = df_psd[df_psd['Frequency band'] == band]
    df_psd_band_final = pd.DataFrame()
    for region in regions:
        df_psd_temp = df_psd_band[[region,'Condition']].copy()
        df_psd_temp.rename(columns={region: band}, inplace=True)
        df_psd_temp['Region'] = region
        df_psd_band_final = pd.concat([df_psd_band_final,df_psd_temp]).reset_index(drop=True)

    plt.figure(dpi=100)
    ax = sns.boxplot(x=x, y=band, hue=hue, data=df_psd_band_final, order=regions,
                     flierprops=dict(markerfacecolor = '0.5', markersize = 3))
    
    pairs = []
    if stat_test=='t-test_paired':
        for i in range(len(condition_comp_list)):
            _,significant = paired_Ttest(df_psd[df_psd['Frequency band']==band],condition_comp_list[i])
            for j in range(len(significant)):
                sign_temp = list(significant[j].values())[0]
                for s in range(len(regions)):
                    if sign_temp == regions[s]:
                        pairs.append(((*significant[j].values(),condition_comp_list[i][0]),(*significant[j].values(),condition_comp_list[i][1])))
    else:
        print(stat_test,'as a parameter for the statistical test is not supported.')

    if len(pairs) != 0:
        annotator = Annotator(ax,pairs=pairs,data=df_psd_band_final, x=x, y=band,
                        hue=hue,plot="boxplot",order=regions)\
                .configure(test=stat_test,text_format='star',loc=ast_loc,text_offset=2)\
                .apply_and_annotate()
    
    plt.legend(title='Condition', title_fontsize=fnt[2],fontsize=fnt[1])
    for i in range(len(condition_legend)):
        ax.legend_.texts[i].set_text(condition_legend[i])

    if ast_loc == 'outside':
        ax.set_title('{} power regional boxplot'.format(band),y=1.125,fontsize=fnt[2])
    else:
        ax.set_title('{} power regional boxplot'.format(band),y=1.025,fontsize=fnt[2])
    
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel('PSD (µV\u00b2/Hz)', fontsize=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}.tiff'.format(band,regions,stat_test),dpi=300,bbox_inches='tight')