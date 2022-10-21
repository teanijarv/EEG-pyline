# ========== Packages ==========
import mne, os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import pandas as pd
from statannotations.Annotator import Annotator
from basic.stats import apply_stat_test

# ========== Functions ==========
def plot_topomaps_band(df_psd_ch,epochs,b_name,condition_legend,conditions=None,
                       fnt=['sans-serif',8,10],cmap=None,title=True,export=False):
    """
    Plot topographical headmaps of power spectra data for the frequency band in interest.

    Parameters
    ----------
    df_psd_ch: A Pandas dataframe of the power spectra values for all the channels.
    epochs: Epochs-type (MNE-Python) EEG file
    b_name: A string for the frequency band in interest
    condition_legend: A list of strings for the experiment conditions plotted
    conditions (optional): A list of strings for experiment conditions codes (takes all if not applied)
    fnt (optional): A list of font, and two font sizes (default: ['sans-serif',8,10])
    cmap (optional): Colormap of the heatmap (Matplotlib)
    title (optional): A boolean for displaying the title of the plot
    export (optional): A boolean for exporting, if True then the plot will be saved (default: False)
    """
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
        im,_ = mne.viz.plot_topomap(ar_psd_meanch_band[i],epochs.info,axes=axs[i],
                                    vmin=vmin,vmax=vmax,show=False,cmap=cmap)
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

    if title == True:
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

def plot_boxplot_location(df_psd,bands,region,condition_comp_list,condition_legend,fnt=['sans-serif',8,10],
    title=True,stat_test='Wilcoxon',ast_loc='inside',ylims=None,flier_label_xyloc=None,annot_offset=[0.1,0.1],
    yscale='linear',legend=False,figsize=(6,4),palette=None,verbose=True,export=False):
    """
    Plot boxplot for PSD values at a location (region or channel) of interest.

    Parameters
    ----------
    df_psd: A Pandas dataframe of the power spectra values (for all the channels or regions)
    bands: A list of strings for all the frequency bands in interest (e.g. ['Delta','Alpha'])
    region: A string for the region/channel in interest (e.g. 'Fp1')
    condition_comp_list: A list of strings for experiment conditions codes to compare (e.g. [['EC_00','EC_06'],['EC_06','EC_07']])
    condition_legend: A list of strings for the experiment conditions plotted
    fnt (optional): A list of font, and two font sizes (default: ['sans-serif',8,10])
    title (optional): A boolean for displaying the title of the plot
    stat_test (optional): A string for the statistical test for comparison (available: 't-test_paired','Wilcoxon')
    ast_loc (optional): A string for placement of asterix for statistical comparison (default: 'inside')
    ylims (optional): A list for y-scale limits (default: None)
    flier_label_xyloc (optional): Custom xy location for outlier label (in case they will be out of plot range)
    annot_offset (optional): Custom offset values for moving statistical test asterix annotations
    yscale (optional): Scale of y-scale (available: 'linear','log')
    legend (optional): Custom legend xy location (Matplotlib)
    figsize (optional): Figure size (Matplotlib)
    palette (optional): Figure color palette (Matplotlib/Seaborn)
    export (optional): A boolean for exporting, if True then the plot will be saved (default: False)
    """
    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    
    x='Frequency band'
    hue='Condition'
    conditions = df_psd['Condition'].unique()

    fliercount = [0,0]
    if ylims != None:
        fliercount[0] = len(df_psd[region].values[df_psd[region] < ylims[0]])
        fliercount[1] = len(df_psd[region].values[df_psd[region] > ylims[1]])

    plt.figure(dpi=100,figsize = figsize)
    ax = sns.boxplot(x=x, y=region, hue=hue, order=bands, data=df_psd,
                     flierprops=dict(markerfacecolor = '0.5', markersize = 1),palette=palette)
    ax.set_yscale(yscale)
    
    pairs = []
    if stat_test=='t-test_paired' or stat_test=='Wilcoxon' or stat_test=='t-test_ind':
        for i in range(len(condition_comp_list)):
            _,_,_,significant = apply_stat_test(df_psd[['Subject','Frequency band','Condition',region]],condition_comp_list[i],stat_test=stat_test,verbose=verbose)
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
                .configure(test=stat_test,loc=ast_loc,text_format='star',verbose=0)\
                .apply_test().annotate(line_offset_to_group=annot_offset[0], line_offset=annot_offset[1])
        
    if legend != False:
        if legend == True:
            kwargs = dict(loc = 'best')
        else:
            kwargs = legend
        plt.legend(title='Condition',title_fontsize=fnt[2],fontsize=fnt[1],**kwargs)
        for i in range(len(condition_legend)):
            ax.legend_.texts[i].set_text(condition_legend[i])
    else:
        ax.get_legend().remove()

    if title == True:
        if ast_loc == 'outside':
            ax.set_title('{} bandpowers boxplot'.format(region),y=1.125,fontsize=fnt[2])
        else:
            ax.set_title('{} bandpowers boxplot'.format(region),y=1.025,fontsize=fnt[2])

    if yscale == 'linear':
        ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel('PSD (µV\u00b2/Hz)', fontsize=fnt[1])

    if ylims != None:
        ax.set(ylim=ylims)
        if flier_label_xyloc == None:
            flier_label_xyloc = [0,0]
            flier_label_xyloc[1] = 2
            if len(bands) == 1:
                flier_label_xyloc[0] = -0.05
            else:
                flier_label_xyloc[0] = (len(bands)-1)/2-0.15
        if fliercount[1] != 0:
            plt.text(flier_label_xyloc[0],ylims[1]-flier_label_xyloc[1],str(fliercount[1])+' outliers \u2191',size=fnt[1])
        if fliercount[0] != 0:
            plt.text(flier_label_xyloc[0],ylims[0]+flier_label_xyloc[1],str(fliercount[0])+' outliers \u2193',size=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}_{}.tiff'.format(region,bands,stat_test,conditions),dpi=300,bbox_inches='tight')
    
def plot_boxplot_band(df_psd,regions,band,condition_comp_list,condition_legend,fnt=['sans-serif',8,10],
    title=True,stat_test='Wilcoxon',ast_loc='inside',ylims=None,flier_label_xyloc=None,annot_offset=[0.1,0.1],
    yscale='linear',legend=False,figsize=(6,4),ylabel='PSD (µV\u00b2/Hz)',palette=None,verbose=True,export=False):
    """
    Plot boxplot for PSD values for a specific frequency band of interest at regions/channels.

    Parameters
    ----------
    df_psd: A Pandas dataframe of the power spectra values (for all the channels or regions)
    regions: A list of strings for all the regions/channels in interest (e.g. ['Fz','Fp1'])
    band: A string for the frequency band in interest (e.g. 'Alpha')
    condition_comp_list: A list of strings for experiment conditions codes to compare (e.g. [['EC_00','EC_06'],['EC_06','EC_07']])
    condition_legend: A list of strings for the experiment conditions plotted
    fnt (optional): A list of font, and two font sizes (default: ['sans-serif',8,10])
    title (optional): A boolean for displaying the title of the plot
    stat_test (optional): A string for the statistical test for comparison (default: 'Wilcoxon')
    ast_loc (optional): A string for placement of asterix for statistical comparison (default: 'inside')
    ylims (optional): A list for y-scale limits (default: None)
    flier_label_xyloc (optional): Custom xy location for outlier label (in case they will be out of plot range)
    annot_offset (optional): Custom offset values for moving statistical test asterix annotations
    yscale (optional): Scale of y-scale (available: 'linear','log')
    legend (optional): Custom legend xy location (Matplotlib)
    figsize (optional): Figure size (Matplotlib)
    palette (optional): Figure color palette (Matplotlib/Seaborn)
    export (optional): A boolean for exporting, if True then the plot will be saved (default: False)
    """
    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    
    x = 'Region'
    hue = 'Condition'
    conditions = df_psd['Condition'].unique()
    
    df_psd_band = df_psd[df_psd['Frequency band'] == band]
    df_psd_band_final = pd.DataFrame()
    for region in regions:
        df_psd_temp = df_psd_band[[region,'Condition']].copy()
        df_psd_temp.rename(columns={region: band}, inplace=True)
        df_psd_temp['Region'] = region
        df_psd_band_final = pd.concat([df_psd_band_final,df_psd_temp]).reset_index(drop=True)
    
    fliercount = [0,0]
    if ylims != None:
        fliercount[0] = len(df_psd_band_final[band].values[df_psd_band_final[band] < ylims[0]])
        fliercount[1] = len(df_psd_band_final[band].values[df_psd_band_final[band] > ylims[1]])

    plt.figure(dpi=100,figsize = figsize)
    ax = sns.boxplot(x=x, y=band, hue=hue, data=df_psd_band_final, order=regions,
                     flierprops=dict(markerfacecolor = '0.5', markersize = 1),palette=palette)
    ax.set_yscale(yscale)

    pairs = []
    if stat_test=='t-test_paired' or stat_test=='Wilcoxon' or stat_test=='t-test_ind':
        for i in range(len(condition_comp_list)):
            _,_,_,significant = apply_stat_test(df_psd[df_psd['Frequency band']==band],condition_comp_list[i],stat_test=stat_test,verbose=verbose)
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
                .configure(test=stat_test,text_format='star',loc=ast_loc,verbose=0)\
                .apply_test().annotate(line_offset_to_group=annot_offset[0], line_offset=annot_offset[1])
    
    if legend != False:
        if legend == True:
            kwargs = dict(loc = 'best')
        else:
            kwargs = legend
        plt.legend(title='Condition',title_fontsize=fnt[2],fontsize=fnt[1],**kwargs)
        for i in range(len(condition_legend)):
            ax.legend_.texts[i].set_text(condition_legend[i])
    else:
        ax.get_legend().remove()
    
    if title == True:
        if ast_loc == 'outside':
            ax.set_title('{} regional boxplot'.format(band),y=1.125,fontsize=fnt[2])
        else:
            ax.set_title('{} regional boxplot'.format(band),y=1.025,fontsize=fnt[2])
    
    if yscale == 'linear':
        ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])
    plt.xlabel(x, fontsize=fnt[1])
    plt.ylabel(ylabel, fontsize=fnt[1])

    if ylims != None:
        ax.set(ylim=ylims)
        if flier_label_xyloc == None:
            flier_label_xyloc = [0,0]
            flier_label_xyloc[1] = 2
            if len(regions) == 1:
                flier_label_xyloc[0] = -0.05
            else:
                flier_label_xyloc[0] = (len(regions)-1)/2-0.15
        if fliercount[1] != 0:
            plt.text(flier_label_xyloc[0],ylims[1]-flier_label_xyloc[1],str(fliercount[1])+' outliers \u2191',size=fnt[1])
        if fliercount[0] != 0:
            plt.text(flier_label_xyloc[0],ylims[0]+flier_label_xyloc[1],str(fliercount[0])+' outliers \u2193',size=fnt[1])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\psdboxplt_{}_{}_{}_{}.tiff'.format(band,regions,stat_test,conditions),dpi=300,bbox_inches='tight')