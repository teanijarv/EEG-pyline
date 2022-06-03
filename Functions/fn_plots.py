import mne, os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd
from statannotations.Annotator import Annotator
from Functions.fn_stats import *

def plot_topomaps_band(df_psd_ch,epochs,b_name,condition_legend,conditions=None,fnt=['sans-serif',8,10],export=False):
    sns.set_style("white",{'font.family': [fnt[0]]})
    
    if conditions == None:
        conditions = df_psd_ch['Condition'].unique()
    
    df_psd_ch_band = df_psd_ch[df_psd_ch['Frequency band'] == b_name].drop(columns=['Frequency band','Subject'])
    ar_psd_meanch_band = [None]*len(conditions)
    vmin = float('inf')
    vmax = 0

    for i in range(len(conditions)):
        ar_psd_meanch_band[i] = df_psd_ch_band[df_psd_ch_band['Condition'] == conditions[i]]\
                                    .drop(columns=['Condition']).to_numpy().mean(axis=0)
        vmin = min([vmin,min(ar_psd_meanch_band[i])])
        vmax = max([vmax,max(ar_psd_meanch_band[i])])
    
    fig,axs = plt.subplots(nrows=1,ncols=len(conditions),dpi=100)
    
    for i in range(len(conditions)):
        im,_ = mne.viz.plot_topomap(ar_psd_meanch_band[i],epochs.info,axes=axs[i],vmin=vmin,vmax=vmax,show=False)
        axs[i].set_title(condition_legend[i],fontsize=fnt[1])

    if len(conditions) <= 2:
        clb_x_start = 0.95
        clb_x_width = 0.04
        clb_y_start = 0.25
        clb_y_height = 0.6
        title_x = 0.525
        title_y = 0.925
    else:
        clb_x_start = 0.95
        clb_x_width = 0.04
        clb_y_start = 0.35
        clb_y_height = 0.4
        title_x = 0.525
        title_y = 0.825

    fig.suptitle('Average PSD across all subjects ({})'.format(b_name),x=title_x,y=title_y,fontsize=fnt[1])
    clb_ax = fig.add_axes([clb_x_start, clb_y_start, clb_x_width, clb_y_height])
    clb = fig.colorbar(im, cax=clb_ax)
    clb.ax.set_ylabel('µV\u00b2/Hz',fontsize=fnt[1])
    clb.ax.tick_params(labelsize=fnt[1])
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdtopo_{}_{}.tiff'.format(b_name,conditions),dpi=300,bbox_inches='tight')

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

    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
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
    
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel('PSD (µV\u00b2/Hz)', fontsize=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}.tiff'.format(band,regions,stat_test),dpi=300,bbox_inches='tight')