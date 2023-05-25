import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.preprocessing import peak_finder

def find_all_peaks(evoked_obj, epochs, t_range=[-200, 800], thresh=None, subject_name='',
                   verbose=False, plot=False):
    time_coef = 1e3
    amplitude_coef= 1e6

    evoked_data = evoked_obj.data[0]
    evoked_time = evoked_obj.times*time_coef

    # Use peak_finder to find only the peaks from the signal
    maxpeak_locs, maxpeak_mags = peak_finder(evoked_data,thresh=None,extrema=1)
    minpeak_locs, minpeak_mags = peak_finder(evoked_data,thresh=None,extrema=-1)

    # Calculate the time from the samples
    maxpeak_locs = (maxpeak_locs/epochs.info['sfreq'])*time_coef-200
    minpeak_locs = (minpeak_locs/epochs.info['sfreq'])*time_coef-200

    # Find the closest value from the signal time array
    minpeak_times = [None]*len(minpeak_locs)
    maxpeak_times = [None]*len(maxpeak_locs)
    for i,loc in enumerate(minpeak_locs):
        difference_array = np.absolute(evoked_time-loc)
        minpeak_times[i] = evoked_time[difference_array.argmin()]
    for i,loc in enumerate(maxpeak_locs):
        difference_array = np.absolute(evoked_time-loc)
        maxpeak_times[i] = evoked_time[difference_array.argmin()]

    # Convert amplitudes to right units
    maxpeak_mags = maxpeak_mags*amplitude_coef
    minpeak_mags = minpeak_mags*amplitude_coef

    if verbose == True:
        print('max peaks\ntimes:',maxpeak_times,'mag:',maxpeak_mags)
        print('min peaks\ntimes:',minpeak_times,'mag:',minpeak_mags)
    if plot == True:
        plt.figure(figsize=(8,3), dpi=100)
        plt.suptitle('All the detected peaks ({})'.format(subject_name))
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.grid(which='major',axis='y',linewidth = 0.15)
        plt.xticks(np.arange(t_range[0], t_range[1], step=100))
        plt.xlim(t_range)
        plt.plot(evoked_time,evoked_data*amplitude_coef,linewidth=0.6,color='black')
        plt.plot(maxpeak_locs, maxpeak_mags, marker='*', linestyle='None', color='r')
        plt.plot(minpeak_locs, minpeak_mags, marker='*', linestyle='None', color='b')
        
        plt.show()

    return minpeak_times,minpeak_mags,maxpeak_times,maxpeak_mags

def identify_erps(evoked_obj, erp_wins, minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags,
                  t_range=[-200, 800], subject_name='', verbose=False, plot=False, savefig=False,
                  results_foldername = "Results/", exp_folder=''):
    # Pre-define variables
    erp_peaks = {}
    not_erp_peaks = {}
    erp_times = []
    erp_mags = []
    all_peak_times = np.concatenate([minpeak_times,maxpeak_times])
    all_peak_mags = np.concatenate([minpeak_mags,maxpeak_mags])
    time_coef = 1e3
    amplitude_coef= 1e6

    evoked_data = evoked_obj.data[0]
    evoked_time = evoked_obj.times*time_coef

    # Check whether any of the pre-defined ERP time windows match with any of the peaks
    for ie, erp in enumerate(erp_wins):
        erp_name = list(erp_wins.keys())[ie]
        tminmax = erp_wins[erp][0:2]
        extremas = erp_wins[erp][-1]
        
        # If the ERP in interest positive, then check in time window and add the peak to ERP dictionary, and vice versa
        if extremas == 1:
            for idx, timepoint in enumerate(maxpeak_times):
                if tminmax[0] <= timepoint <= tminmax[1]:
                    if erp_name not in list(erp_peaks.keys()):
                        erp_peaks[erp_name] = [timepoint,maxpeak_mags[idx]]
                        erp_times.append(timepoint)
                        erp_mags.append(maxpeak_mags[idx])
                    else:
                        if maxpeak_mags[idx] > erp_peaks[erp_name][1]:
                            erp_peaks[erp_name] = [timepoint,maxpeak_mags[idx]]
                            erp_times.append(timepoint)
                            erp_mags.append(maxpeak_mags[idx])
        else:
            for idx, timepoint in enumerate(minpeak_times):
                if tminmax[0] <= timepoint <= tminmax[1]:
                    if erp_name not in list(erp_peaks.keys()):
                        erp_peaks[erp_name] = [timepoint,minpeak_mags[idx]]
                        erp_times.append(timepoint)
                        erp_mags.append(minpeak_mags[idx])
                    else:
                        if minpeak_mags[idx] < erp_peaks[erp_name][1]:
                            erp_peaks[erp_name] = [timepoint,minpeak_mags[idx]]
                            erp_times.append(timepoint)
                            erp_mags.append(minpeak_mags[idx])

    # Create a dictionary for all the peaks which did not get classified as ERPs
    not_erp_times = list(set(all_peak_times) - set(erp_times))
    not_erp_mags = list(set(all_peak_mags) - set(erp_mags))
    for idx in range(len(not_erp_times)):
        not_erp_peaks['N/A peak '+str(idx)] = [not_erp_times[idx],not_erp_mags[idx]]

    if verbose == True:
        print('ERPs\n',erp_peaks)
        print('Other peaks\n',not_erp_peaks)

    if plot == True:
        plt.figure(figsize=(8,3), dpi=100)
        plt.suptitle('Event-related potentials ({})'.format(subject_name))
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.plot(evoked_time,evoked_data*amplitude_coef,linewidth=0.6,color='black')
        plt.grid(which='major',axis='y',linewidth = 0.15)
        plt.xticks(np.arange(t_range[0], t_range[1], step=100))
        plt.xlim(t_range)
        for ie, erp_name in enumerate(erp_peaks):
            #print(ie,erp_name,erp_peaks[erp_name])
            if 'N' in erp_name:
                color = 'b'
            elif 'P' in erp_name:
                color = 'r'
            else:
                color = 'g'
            plt.plot(erp_peaks[erp_name][0], erp_peaks[erp_name][1], marker='*', linestyle='None', color=color)
            plt.annotate(erp_name, (erp_peaks[erp_name][0]+15,erp_peaks[erp_name][1]-0.15))
        if savefig == True:
            plt.savefig(fname='{}/{}/ERP analysis/{}_erpfig.png'.format(results_foldername,
                                                                        exp_folder, subject_name)) # add ERP plots to precreation function
        plt.show()
    
    return erp_peaks, not_erp_peaks

def find_minmax_erp(evoked_obj, erp_peaks, erp_tochange, new_time_win, t_range=[-200, 800],
                    subject_name='', verbose=False, plot=False, savefig=False,
                    results_foldername = "Results/", exp_folder=''):
    time_coef = 1e3
    amplitude_coef= 1e6

    evoked_data = evoked_obj.data[0]*amplitude_coef
    evoked_time = evoked_obj.times*time_coef

    # Check only time values within the time window
    timewin_idx = np.where(np.logical_and(evoked_time>=new_time_win[0],evoked_time<=new_time_win[1]))
    evoked_data_win = evoked_data[timewin_idx]
    evoked_time_win = evoked_time[timewin_idx]
    
    try:
        if new_time_win[2] == 1:
            peak_idx = np.argmax(evoked_data_win)
        elif new_time_win[2] == -1:
            peak_idx = np.argmin(evoked_data_win)
        else:
            print('Did not choose to check whether positive or negative values (1 - max; -1 - min).')
            return erp_peaks
    except:
        print('Defined time window parameters are not valid.')
        return erp_peaks

    lat = evoked_time_win[peak_idx]
    amp = evoked_data_win[peak_idx]

    erp_peaks[erp_tochange] = [lat,amp]

    if verbose == True:
        print('ERPs\n',erp_peaks)

    if plot == True:
        plt.figure(figsize=(8,3), dpi=100)
        plt.suptitle('Event-related potentials ({})'.format(subject_name))
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.plot(evoked_time,evoked_data,linewidth=0.6,color='black')
        plt.grid(which='major',axis='y',linewidth = 0.15)
        plt.xticks(np.arange(t_range[0], t_range[1], step=100))
        plt.xlim(t_range)
        for ie, erp_name in enumerate(erp_peaks):
            #print(ie,erp_name,erp_peaks[erp_name])
            if 'N' in erp_name:
                color = 'b'
            elif 'P' in erp_name:
                color = 'r'
            else:
                color = 'g'
            plt.plot(erp_peaks[erp_name][0], erp_peaks[erp_name][1], marker='*', linestyle='None', color=color)
            plt.annotate(erp_name, (erp_peaks[erp_name][0]+15,erp_peaks[erp_name][1]-0.15))
        if savefig == True:
            plt.savefig(fname='{}/{}/ERP analysis/{}_erpfig.png'.format(results_foldername,
                                                                        exp_folder, subject_name)) # add ERP plots to precreation function
        plt.show()

    return erp_peaks

def erp_dict_to_df(erp_peaks,erp_wins,subject_name):
    df = pd.DataFrame(erp_peaks)
    df_erps_lat = pd.DataFrame(index=list(erp_wins.keys()))
    df_erps_amp = pd.DataFrame(index=list(erp_wins.keys()))

    df_erps_lat[subject_name] = df.loc[0]
    df_erps_amp[subject_name] = df.loc[1]

    df_erps_lat = df_erps_lat.T.add_suffix(' latency')
    df_erps_amp = df_erps_amp.T.add_suffix(' amplitude')

    df_erps = pd.concat([df_erps_lat,df_erps_amp],axis=1)

    return df_erps