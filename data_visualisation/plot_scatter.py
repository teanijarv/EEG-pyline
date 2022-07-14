# ========== Packages ==========
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ========== Functions ==========
def plot_correlation(df_psd_clinical_comparison,band,comparison_cond,region,clinical_outcome,state,labels,
                      correlation='Spearman',fnt=['sans-serif',9,8],color_palette=[None,None],title=True,export=False):
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
    plt.figure(dpi=100)
    if len(state) == 1:
        ax = sns.scatterplot(data=data, x=clinical_outcome, y=region, color=color_palette[0])
        ax.annotate(f'$\\rho = {r[0]:.3f}, p = {p[0]:.3f}$',
                        xy=(0.675, 1.025), xycoords='axes fraction', size=fnt[1])
    elif len(state) == 2:
        ax = sns.scatterplot(data=data, x=clinical_outcome, y=region, palette=color_palette[1], hue='State')
        ax.annotate(f'$\\rho_c = {r[0]:.3f}, p_c = {p[0]:.3f}$',
                        xy=(0.675, 1.1), xycoords='axes fraction', size=fnt[1])
        ax.annotate(f'$\\rho_o = {r[1]:.3f}, p_o = {p[1]:.3f}$',
                        xy=(0.675, 1.025), xycoords='axes fraction', size=fnt[1])
        plt.legend(title='State',title_fontsize=fnt[2],fontsize=fnt[1],**{'loc':'upper right','bbox_to_anchor':(0.255, 1.225)})
    if correlation == 'Spearman':
        ax.set(xlim=(0, 101),ylim=(0, 102))
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
