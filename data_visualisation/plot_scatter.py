# ========== Packages ==========
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ========== Functions ==========
def plot_correlation(df_psd_clinical_comparison,band,comparison_cond,region,clinical_outcome,state,labels,
                      correlation='Spearman',fnt=['sans-serif',9,8],color_palette=[None,None],title=True,
                      trend_line_ci=[False,None],legend=True,export=False):
    """
    Plot scatter plot for correlation between PSD values and clinical outcomes at a location and band of interest.

    Parameters
    ----------
    df_psd_clinical_comparison: A Pandas dataframe of PSD values and clinical outcomes compared between timepoints
    band: The frequency band of the PSD values (e.g. 'Alpha')
    comparison_cond: A list of strings for conditions to compare (e.g. ['06','07'])
    region: A string for the region/channel in interest (e.g. 'Fp1')
    clinical_outcome: A string for the clinical outcome in interest (e.g. 'Dass21_depression')
    state: A list of strings for the states of interest (e.g. ['Eyes closed','Eyes open'])
    labels: A list of strings for x and y labels (e.g. ['DASS-D (%)','Theta PSD at occipital region (%)'])
    correlation (optional): A string of the correlation test (available: 'Spearman','Pearson')
    fnt (optional): A list of font, and two font sizes (default: ['sans-serif',8,10])
    color_palette (optional): Figure color palette (Matplotlib/Seaborn) (e.g. ["#5A9","husl"])
    title (optional): A boolean for displaying the title of the plot
    export (optional): A boolean for exporting, if True then the plot will be saved (default: False)
    """
    
    data = df_psd_clinical_comparison[df_psd_clinical_comparison['Frequency band']==band]
    data = data[data['Condition']==str(comparison_cond)]

    r, p = [None]*len(state), [None]*len(state)
    if correlation == 'Spearman':
        if len(state) == 1:
            data = data[data['State']==state[0]]
            r[0],p[0] = stats.spearmanr(data[data['State']==state[0]][clinical_outcome], data[data['State']==state[0]][region]) 
        elif len(state) == 2:
            r[0], p[0] = stats.spearmanr(data[data['State']==state[0]][clinical_outcome], data[data['State']==state[0]][region])
            r[1], p[1] = stats.spearmanr(data[data['State']==state[1]][clinical_outcome], data[data['State']==state[1]][region])
        else:
            print('Can only plot 1 or 2 states.')
        
        data_ranked = data.rank(numeric_only=True,pct=True)*100
        data_ranked = pd.concat([data_ranked,data[['Condition','Frequency band','State']]],axis=1)
        data = data_ranked

    if correlation == 'Pearson':
        if len(state) == 1:
            data = data[data['State']==state[0]]
            r[0],p[0] = stats.pearsonr(data[data['State']==state[0]][clinical_outcome], data[data['State']==state[0]][region]) 
        elif len(state) == 2:
            r[0], p[0] = stats.pearsonr(data[data['State']==state[0]][clinical_outcome], data[data['State']==state[0]][region])
            r[1], p[1] = stats.pearsonr(data[data['State']==state[1]][clinical_outcome], data[data['State']==state[1]][region])
        else:
            print('Can only plot 1 or 2 states.')

    sns.set_style("whitegrid",{'font.family': [fnt[0]]})
    plt.figure(dpi=300)
    if len(state) == 1:
        ax = sns.lmplot(data=data, x=clinical_outcome, y=region, scatter_kws = {'color': color_palette[0]},
                        line_kws = {'color': color_palette[0]}, fit_reg = trend_line_ci[0], ci = trend_line_ci[1])
        plt.annotate(f'r = {r[0]:.3f}, p = {p[0]:.3f}',
                        xy=(0, 1.025), xycoords='axes fraction', size=fnt[1])
    elif len(state) == 2:
        ax = sns.lmplot(data=data, x=clinical_outcome, y=region, palette=color_palette, hue='State',
                        fit_reg = trend_line_ci[0], ci = trend_line_ci[1], legend = legend)
        plt.annotate(f'EC: r = {r[0]:.3f}, p = {p[0]:.3f}',
                        xy=(0, 1.08), xycoords='axes fraction', size=fnt[1])
        plt.annotate(f'EO: r = {r[1]:.3f}, p = {p[1]:.3f}',
                        xy=(0, 1.025), xycoords='axes fraction', size=fnt[1])
        #plt.legend(title='State',title_fontsize=fnt[2],fontsize=fnt[1],**{'loc':'upper right','bbox_to_anchor':(1.255, 1.225)})
    if correlation == 'Spearman':
        ax.set(xlim=(0, 101),ylim=(0, 101))
    if title == True:
        plt.suptitle("{}'s correlation at {} region".format(correlation,region))
    plt.xlabel(labels[0],fontsize=fnt[1])
    plt.ylabel(labels[1],fontsize=fnt[1])
    plt.tick_params(axis='both', which='major', labelsize=fnt[2])

    if export == True:
        try:
            os.makedirs(r"Results")
        except FileExistsError:
            pass
        plt.savefig('Results\{}corscatter_{}_{}_{}_{}_{}.tiff'.format(correlation,region,band,clinical_outcome,comparison_cond,state),dpi=300,bbox_inches='tight')
