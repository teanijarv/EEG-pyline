# pyline_analysis.py
import os 
import warnings
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from mne.preprocessing import peak_finder
import neurokit2 as nk

# Import PylineManage from the pyline package
from . import PylineManage
manage = PylineManage ()

class PylineAnalysis:
    def __init__(self):
        """
        Initialize EEG Analysis.

        Parameters:
        - 
        """

    def equalize_event_counts(self, data, condition_comp):
        """
        Equalize event counts in epochs.

        Parameters:
        - condition_comp (list of str): List of conditions for which event counts should be equalized.

        Returns:
        - equal_epochs (mne.Epochs): Equalized epochs object.
        """
        if not isinstance(data, mne.epochs.Epochs):
            raise ValueError("Data must be an instance of mne Epochs.")
        
        equal_epochs = data.copy()
        equal_epochs = equal_epochs.equalize_event_counts(condition_comp)
        return equal_epochs
    
    def create_evoked (self, data, channel = None):
        """
        Create evoked responses.

        Returns:
        - evoked_dict (dict): Dictionary containing evoked responses.
        """
        if not isinstance(data, (mne.epochs.Epochs, mne.epochs.EpochsArray)):
            raise ValueError("Data must be an instance of mne Epochs.")
        
        if channel:
            evoked = data.copy().pick_channels([channel]).average()
            print("Evoked object created for {} epochs only".format(channel))
        else:
            evoked = data.copy().average()
            print("Evoked object created across all epochs")
        return evoked

    def evoked_compare(self, data, comparator):
        """
        Plot evoked time series for two conditions. These are saved in list prior to passing to the plotting function. 
        Condition abels are taken from the .comment attributes of each evoked object.

        Parameters:
        - data: MNE Evoked object
        - comparator: MNE Evoked object
        """
        
        compare = [data, comparator]

        fig = mne.viz.plot_compare_evokeds(compare,
                                            legend="upper left",
                                            show_sensors="upper right")

        return fig

    def epoch_to_array(self, data, existing_array=None):
        """
        Concatenate the data array of an MNE Epochs object to an existing array along the epochs axis.
        
        Args:
            epochs (mne.Epochs): The Epochs object.
            existing_array (numpy.ndarray, optional): An existing 3D array to concatenate to.
                If None, the Epochs data array will be returned.
            
        Returns:
            numpy.ndarray: The concatenated data array with shape (n_epochs, n_channels, n_times).
        """
        if not isinstance(data, (mne.epochs.Epochs, mne.epochs.EpochsArray)):
            raise ValueError("Data must be an instance of mne Epochs.")
        
        # Get the data array from the Epochs object
        epoch_array = data.get_data(picks=['eeg'])
        
        if existing_array is None:
            # Return the Epochs data array
            combined_array = epoch_array
        else:
            # Concatenate the arrays along the epochs axis (axis=0)
            combined_array = np.concatenate((existing_array, epoch_array), axis=0)
        
        return combined_array

    def calculate_dprime(self, H, M, FA, CR):
        """
        Calculate d-prime (d') given counts of hits, misses, false alarms, and correct rejections.
        
        Args:
        - H (int): Number of hits (correct detections of target stimuli).
        - M (int): Number of misses (failure to detect target stimuli).
        - FA (int): Number of false alarms (incorrect detections of non-target stimuli as target).
        - CR (int): Number of correct rejections (correct identification of non-target stimuli).
        
        Returns:
        - d_prime (float): Calculated d-prime value.
        """
        # Ensure no division by zero
        if H == 0:
            H += 0.5  # Add a small value to H to avoid division by zero
        if FA == 0:
            FA += 0.5  # Add a small value to FA to avoid division by zero

        # Adjust HR and FAR if they are 0 or 1
        epsilon = 0.01  # Small value to avoid issues with norm.ppf

        HR = H / (H + M)  # Hit rate
        if HR == 1:
            HR = 1 - epsilon

        FAR = FA / (FA + CR)  # False alarm rate
        if FAR == 0:
            FAR = epsilon
        elif FAR == 1:
            FAR = 1 - epsilon
        
        # Calculate d-prime using the formula: d' = Z(HR) - Z(FAR)
        Z_HR = norm.ppf(HR)  # Inverse of the cumulative distribution function (CDF) for hit rate
        Z_FAR = norm.ppf(FAR)  # Inverse of the cumulative distribution function (CDF) for false alarm rate
        
        d_prime = Z_HR - Z_FAR
        
        return d_prime

    def extract_trial_data(self, dataframe, data, bids, sub, params, task = None):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame to store the results.
            data (object): Object containing event data.
            sub (int): current index of for loop.
            eeg_params (object): Object containing EEG parameters.
            cue (str, optional): Label of the cue event. Defaults to none.
            target (str): Label of the target event.

        Returns:
            pandas.DataFrame: DataFrame containing counts of successful trials and average reaction times.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame")
        
        if data.events is None:
            raise ValueError("data.events is None")
        
        if task is None:
            raise ValueError('Task must be specified in params.event_dict')
        
        # add in checks for target in task saved in params object

        temp_dataframe = pd.DataFrame()

        if task == 'AXCPT':

            AX_success_events = []
            AX_success_RT = []
            BX_commission_events = []
            BX_commission_RT = []
            AB_commission_events = [] # A* really
            AB_commission_RT = [] # A* really

            print('\n Counting trials for {}: {} - {} ({} / {})'.format(task, bids[sub]['subject'], bids[sub]['session'], sub, len(bids)))

            # Calculate total target (AX), non-target (BX/*X), A* (AB), B and total of each (for calculating relative %s).
            total_AX = np.sum(data.events[:, 2] == params.event_dict[task]['AX'])
            total_BX = np.sum(data.events[:, 2] == params.event_dict[task]['*X'])
            total_AB = np.sum(data.events[:, 2] == params.event_dict[task]['Filler/4']) - total_AX # A*
            total_B = np.sum(data.events[:, 2] == params.event_dict[task]['Filler/8']) - total_BX # B's minus BX combinations
            total = total_AX + total_BX  + total_AB + total_B

            # Create an array and dataframe of successful (followed with button press) and unsuccessful A-X trials (no button press)

            for m in range(len(data.events) - 2):  # subtract 2 to avoid index out of bounds

                if (data.events[m][2] == params.event_dict[task]['Filler/4'] and
                        data.events[m+1][2] == params.event_dict[task]['AX'] and
                        data.events[m+2][2] == params.event_dict[task]['button_id']):
                    
                    AX_success_events.append(data.events[m][0])
                    AX_success_RT.append((data.events[m+2][0] - data.events[m+1][0])/data.raw.info['sfreq'])  # calculate reaction time (use sfreq stored in eeg_params)

                elif (data.events[m][2] == params.event_dict[task]['Filler/8'] and
                        data.events[m+1][2] == params.event_dict[task]['*X'] and
                        data.events[m+2][2] == params.event_dict[task]['button_id']):
                    
                    BX_commission_events.append(data.events[m][0])
                    BX_commission_RT.append((data.events[m+2][0] - data.events[m+1][0])/data.raw.info['sfreq'])  # calculate reaction time (use sfreq stored in eeg_params)

                elif (data.events[m][2] == params.event_dict[task]['Filler/4'] and 
                        data.events[m+1][2] == params.event_dict[task]['Filler/8'] and
                        data.events[m+2][2] == params.event_dict[task]['button_id']):
                    
                    AB_commission_events.append(data.events[m][0])
                    AB_commission_RT.append((data.events[m+2][0] - data.events[m+1][0])/data.raw.info['sfreq'])  # calculate reaction time (use sfreq stored in eeg_params)
        
            # Convert each list to a numpy array
            AX_success_events = np.asarray(AX_success_events)
            AX_success_RT = np.asarray(AX_success_RT)
            BX_commission_events = np.asarray(BX_commission_events)
            BX_commission_RT = np.asarray(BX_commission_RT)
            AB_commission_events = np.asarray(AB_commission_events)
            AB_commission_RT = np.asarray(AB_commission_RT)

            # Calculate variables for d-prime
            hits = len(AX_success_events)  # Number of AX correct responses
            misses = total_AX - hits  # Number of AX response misses
            BX_false_alarms = len(BX_commission_events)  # Number of BX false alarms (i.e. responses)
            BX_correct_rejections = total_BX - BX_false_alarms  # Number of BX correct rejections (i.e. non-responses)
            AB_false_alarms = len(AB_commission_events)  # Number of BX false alarms (i.e. responses)
            AB_correct_rejections = total_AB - AB_false_alarms  # Number of BX correct rejections (i.e. non-responses)

            d_prime_AX_BX = self.calculate_dprime(hits, misses, BX_false_alarms, BX_correct_rejections)
            d_prime_AX_AB = self.calculate_dprime(hits, misses, AB_false_alarms, AB_correct_rejections)

            # Calculate probabilities
            prob_a = (total_AX + total_AB)/total # P(A)
            prob_x = (total_AX + total_BX)/total # P(X)
            prob_x_given_a = total_AX/(total_AX + total_AB) # P(X|A)
            prob_x_given_a_over_marginal = prob_x_given_a/prob_a # P(X|A)/P(A)

            # Convert to np.array
            temp_dataframe = pd.DataFrame({
                'Timepoint': bids[sub]['session'],
                'Task': task,
                'Group': bids[sub]['group'],
                'Total AX': total_AX,
                'AX successes': hits,
                'AX omissions': misses,
                'Total BX': total_BX,
                'BX successes': BX_correct_rejections,
                'BX commissions': BX_false_alarms,
                'd-prime (AX-BX)': d_prime_AX_BX,
                'Total A*': total_AB, 
                'A* successes': AB_correct_rejections,
                'A* commissions': AB_false_alarms, 
                'd-prime (AX-AB)': d_prime_AX_AB,
                'Total B': total_B,
                'P(A)': prob_a,
                'P(X)': prob_x,
                'P(X|A)': prob_x_given_a,
                'P(X|A)/P(A)': prob_x_given_a_over_marginal,
                'Median_AX_RT': np.median(AX_success_RT) if len(AX_success_RT) > 0 else 0,
                'Median_BX_RT': np.median(BX_commission_RT) if len(BX_commission_RT) > 0 else 0,
                'Median_A*_RT': np.median(AB_commission_RT) if len(AB_commission_RT) > 0 else 0,
            }, index=[bids[sub]['subject']])

            dataframe = pd.concat([dataframe, temp_dataframe])

            return dataframe
            
        elif task == 'AO':

            success_events = []
            success_RT = []
            commission_events = []
            commission_RT = []

            print('\n Counting trials for {}: {} - {} ({} / {})'.format(task, bids[sub]['subject'], bids[sub]['session'], sub, len(bids)))

            total_targets = 0
            total_nontargets = np.sum(data.events[:, 2] == 2)

            for target in params.event_dict['AO'].keys():
                if target == 'button_id':
                    continue

                total_targets = total_targets + np.sum(data.events[:, 2] == params.event_dict[task][target])
                
                for m in range(len(data.events) - 1):  # subtract 1 to avoid index out of bounds

                    if (data.events[m][2] == params.event_dict[task][target] and
                            data.events[m+1][2] == params.event_dict[task]['button_id']): # store button id in event dict for that task
                        success_events.append(data.events[m][0])
                        success_RT.append((data.events[m+1][0] - data.events[m][0])/data.raw.info['sfreq'])  # calculate reaction time (use sfreq stored in eeg_params)

            # Commissions counted separately so as to not double count
            for m in range(len(data.events) - 1):  # subtract 1 to avoid index out of bounds
                if (data.events[m][2] == 2 and data.events[m+1][2] == params.event_dict[task]['button_id']): 
                    commission_events.append(data.events[m][0])
                    commission_RT.append((data.events[m+1][0] - data.events[m][0])/data.raw.info['sfreq'])

            # Convert to np.array
            success_events = np.asarray(success_events)
            success_RT = np.asarray(success_RT)
            commission_events = np.asarray(commission_events)
            commission_RT = np.asarray(commission_RT)

            # Calculate variables for d-prime
            hits = len(success_events)  # Total number of hits (i.e. target responses)
            misses = total_targets - len(success_events)  # Total number of misses (i.e. target nonresponses)
            false_alarms = len(commission_events)  # Number of false alarms (i.e. standard responses)
            correct_rejections = total_nontargets - len(commission_events)  # Number of correct rejections (i.e. standard nonresponses)

            d_prime = self.calculate_dprime(hits, misses, false_alarms, correct_rejections)

            # Convert to np.array
            temp_dataframe = pd.DataFrame({
                'Timepoint': bids[sub]['session'],
                'Task': task,
                'Group': bids[sub]['group'],
                'Total targets': total_targets,
                'Target successes': hits,
                'Omissions': misses,
                'Total standard': total_nontargets,
                'Standard successes': correct_rejections,
                'Commissions': false_alarms,
                'd-prime': d_prime,
                'P(target)': total_targets /(total_targets + total_nontargets),
                'P(standard)': total_nontargets /(total_targets + total_nontargets),
                'Avg_Successful_RT': np.median(success_RT) if len(success_RT) > 0 else 0,
                'Avg_Commission_RT': np.median(commission_RT) if len(commission_RT) > 0 else 0
            }, index=[bids[sub]['subject']])

            dataframe = pd.concat([dataframe, temp_dataframe])

            return dataframe

    def export(self, data, type = None, outputdir = None, filename = None): 
        
        if type is None or filename is None:
            raise ValueError("Export type and filename must be specified")
                    
        if data is None:
            raise ValueError("Data must be parsed")

        if outputdir is None:
            raise ValueError("You must specify an output directory")

        os.makedirs(outputdir, exist_ok=True)

        full_filename = os.path.join(outputdir, filename)

        if type == 'array':
            if not isinstance(data, (mne.epochs.Epochs, np.ndarray)):
                raise ValueError("For type 'array' an mne epochs object or numpy array must be parsed")
            if isinstance(data, mne.epochs.Epochs):
                export_file = self.epoch_to_array(data, existing_array=None)
                np.save(full_filename, export_file)
            elif isinstance(data, np.ndarray):
                export_file = data
                np.save(full_filename, export_file)

        elif type == 'csv':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("For type 'csv' a pandas dataframe must be parsed")
            filename = filename + '.' + type
            full_filename = os.path.join(outputdir, filename)
            data.to_csv(full_filename)

        elif type == 'xlsx':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("For type 'xlsx' a pandas dataframe must be parsed")
            filename = filename + '.' + type
            full_filename = os.path.join(outputdir, filename)
            data.to_excel(full_filename, index=False)

        elif type == 'set' or type == 'edf':
            if not isinstance(data, (mne.io.edf.edf.RawEDF)):
                raise ValueError("For type 'set/edf' an mne raw object instance must be parsed")
            if isinstance(data, (mne.io.edf.edf.RawEDF)):
                filename = filename + '_raw.' + type
                full_filename = os.path.join(outputdir, filename)
                mne.io.edf.edf.RawEDF.export(data, fname=full_filename, overwrite=True)
                
        elif type == 'fif':
            if not isinstance(data, ( mne.epochs.Epochs, mne.epochs.EpochsArray, mne.evoked.Evoked)):
                raise ValueError("For type 'fif' an mne epochs/evoked object must be parsed")
            if isinstance(data, mne.epochs.Epochs):
                filename = filename + '-epo.' + type
                full_filename = os.path.join(outputdir, filename)
                mne.Epochs.save(data, fname=full_filename, fmt='double', overwrite = True)
            elif isinstance(data, mne.evoked.Evoked):
                filename = filename + '-ave.' + type
                full_filename = os.path.join(outputdir, filename)
                mne.Evoked.save(data, fname=full_filename, overwrite=True)
    
    def find_erp_peaks(self, evoked, epochs, t_range=[-200, 1000], thresh=None, subject_name=None, verbose=False, plot=False):
        """
        Find all the peaks in the evoked signal.

        Args:
            evoked (numpy.ndarray): The evoked signal.
            epochs (numpy.ndarray): The epochs data.
            t_range (list, optional): The time range to consider for peak detection. Defaults to [-200, 1000].
            thresh (float, optional): The threshold for peak detection. Defaults to None.
            subject_name (str, optional): The name of the subject. Defaults to 'Test'.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            plot (bool, optional): Whether to plot the results. Defaults to False.

        Returns:
            tuple: Minimum peak times and magnitudes, maximum peak times and magnitudes.
        """
        time_coef = 1e3
        amplitude_coef= 1e6

        evoked_data = evoked.data[0]
        evoked_time = evoked.times*time_coef

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

        return minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags

        # return erpan.find_all_peaks(evoked, epochs, t_range=t_range, thresh=thresh, subject_name=subject_name, verbose=verbose, plot=plot)

    def identify_erps(self, evoked, erp_wins, minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags, t_range=[-200, 1000], subject_name='Test', verbose=False, plot=True, savefig=False, results_foldername=None, exp_folder=None):
        """
        Identify which peaks are which ERPs based on the pre-defined ERP time windows.

        Args:
            evoked (numpy.ndarray): The evoked signal.
            erp_wins_temp (dict): The temporary ERP time window parameters.
            minpeak_times (list): The minimum peak times.
            minpeak_mags (list): The minimum peak magnitudes.
            maxpeak_times (list): The maximum peak times.
            maxpeak_mags (list): The maximum peak magnitudes.
            t_range (list, optional): The time range to consider for ERP identification. Defaults to [-200, 1000].
            subject_name (str, optional): The name of the subject. Defaults to 'Test'.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            savefig (bool, optional): Whether to save the plot. Defaults to False.
            results_foldername (str, optional): The folder name for saving the results.
            exp_folder (str, optional): The experiment folder name.

        Returns:
            tuple: ERP peaks and non-ERP peaks.
        """
        # Pre-define variables
        erp_peaks = {}
        not_erp_peaks = {}
        erp_times = []
        erp_mags = []
        all_peak_times = np.concatenate([minpeak_times,maxpeak_times])
        all_peak_mags = np.concatenate([minpeak_mags,maxpeak_mags])
        time_coef = 1e3
        amplitude_coef= 1e6

        evoked_data = evoked.data[0]
        evoked_time = evoked.times*time_coef

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
    
        # return erpan.identify_erps(evoked, erp_wins_temp, minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags, t_range=t_range, subject_name=subject_name, verbose=verbose, plot=plot, savefig=savefig, results_foldername=results_foldername, exp_folder=exp_folder)

    def find_minmax_erp(self, evoked, erp_peaks, erp_tochange, new_time_win, t_range=[-200, 1000], subject_name=None, verbose=False, plot=True, savefig=False, results_folder=None, exp_folder=None):
        """
        Find the minimum or maximum value of an ERP within the specified time window.

        Args:
            evoked (numpy.ndarray): The evoked signal.
            erp_peaks (dict): The ERP peaks.
            erp_tochange (str): The name of the ERP to change.
            new_time_win (list): The new time window parameters.
            t_range (list, optional): The time range to consider. Defaults to [-200, 1000].
            subject_name (str, optional): The name of the subject. Defaults to 'Test'.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            plot (bool, optional): Whether to plot the results. Defaults to True
        """
        time_coef = 1e3
        amplitude_coef= 1e6

        evoked_data = evoked.data[0]*amplitude_coef
        evoked_time = evoked.times*time_coef

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
                plt.savefig(fname='{}/{}/ERP analysis/{}_erpfig.png'.format(results_folder,
                                                                            exp_folder, subject_name)) # add ERP plots to precreation function
            plt.show()

        return erp_peaks
        # return erpan.find_minmax_erp(evoked, erp_peaks, erp_tochange, new_time_win,
        #                                     t_range=t_range, subject_name=subject_name, verbose=verbose, plot=plot,
        #                                     savefig=savefig, results_foldername=results_folder, exp_folder=exp_folder)

    def erp_dict_to_df(self, erp_peaks, erp_wins_temp, subject_name, session, group, event=None, channel=None):
        """
        Convert the ERP peaks dictionary to a pandas DataFrame.

        Args:
            erp_peaks (dict): The dictionary of ERP peaks.
            erp_wins_temp (dict): The temporary ERP time window parameters.
            subject_name (str): The name of the subject.
            event (str, optional): The event name. Defaults to None.
            channel (str, optional): The channel name. Defaults to None.

        Returns:
            pandas.DataFrame: The ERP analysis results.
        """
        df_erps_temp = pd.DataFrame()

        for erp_name, erp_data in erp_peaks.items():
            df_erps_temp = df_erps_temp.append({
                'Subject': subject_name,
                'Session': session,
                'Group': group,
                'Event': event,
                'Channel': channel,
                'ERP': erp_name,
                'Time Window': str(erp_wins_temp[erp_name]),
                'Latency': erp_data[0],
                'Amplitude': erp_data[1]
            }, ignore_index=True)

        return df_erps_temp

    def calculate_relative_amplitudes(self, df_erps):
        """
        Calculate the relative peak-to-peak amplitudes between the ERPs.

        Args:
            df_erps (pandas.DataFrame): The ERP analysis results.

        Returns:
            pandas.DataFrame: The ERP analysis results with relative amplitudes.
        """
        print('Adding relative amplitudes for N1-P2, P2-N2, N2-P3')
        #N1_amplitude = df_erps[df_erps['ERP'] == 'N1']['Amplitude'].values[0] if not df_erps[df_erps['ERP'] == 'N1']['Amplitude'].isna().any() else np.nan
        N2_amplitude = df_erps[df_erps['ERP'] == 'N2']['Amplitude'].values[0] if not df_erps[df_erps['ERP'] == 'N2']['Amplitude'].isna().any() else np.nan
        P2_amplitude = df_erps[df_erps['ERP'] == 'P2']['Amplitude'].values[0] if not df_erps[df_erps['ERP'] == 'P2']['Amplitude'].isna().any() else np.nan
        P3_amplitude = df_erps[df_erps['ERP'] == 'P3']['Amplitude'].values[0] if not df_erps[df_erps['ERP'] == 'P3']['Amplitude'].isna().any() else np.nan

        #df_erps['N1_P2_diff'] = np.nan if np.isnan(N1_amplitude) or np.isnan(P2_amplitude) else P2_amplitude - N1_amplitude
        df_erps['P2_N2_diff'] = np.nan if np.isnan(P2_amplitude) or np.isnan(N2_amplitude) else N2_amplitude - P2_amplitude
        df_erps['N2_P3_diff'] = np.nan if np.isnan(N2_amplitude) or np.isnan(P3_amplitude) else P3_amplitude - N2_amplitude
        return df_erps

    def erp_analysis(self, evoked, epochs, params, auto = False, verbose = False, master_erp_df=None, plot=False, subject='Test', session = 'Test', group = 'N/A', event=None, channel=None, results_folder=None, exp_folder=None):
        # Check if necessary arguments are provided
        if event is None or channel is None or params.erp_windows is None:
            raise ValueError("Event, Channel, or ERP windows not specified")
            
        # Check if master_erp_df is provided and is a DataFrame
        if master_erp_df is None or not isinstance(master_erp_df, pd.DataFrame):
            raise ValueError("Invalid DataFrame provided for storing results. Must be pandas Dataframe")
        
        # Find all the peaks in the evoked signal
        minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags = self.find_erp_peaks(evoked, epochs, plot=plot, subject_name=subject, verbose=verbose)

        # Identify which peaks are which ERPs based on the pre-defined ERP time windows
        erp_peaks, not_erp_peaks = self.identify_erps(evoked, params.erp_windows, minpeak_times, minpeak_mags, maxpeak_times, maxpeak_mags,
                                                      subject_name=subject, plot=plot, verbose = verbose, 
                                                      savefig=False, results_foldername=results_folder, exp_folder=exp_folder)

        if not auto:
            # Allow manual time window changes
            while input('Do you need to do any manual time window changes? (leave empty if "no")') != '':
                print('Changing time window parameters for {}'.format(subject))
                new_time_win = [None, None, None]

                # Ask user for which ERP they want to change or add
                erp_tochange = input('What ERP time window you want to change (e.g., N1)?')

                # Ask user for new time window parameters
                new_time_win[0] = int(input('Enter MIN time of the window in interest for {} (e.g., 50)'.format(erp_tochange)))
                new_time_win[1] = int(input('Enter MAX time of the window in interest for {} (e.g., 100)'.format(erp_tochange)))
                new_time_win[2] = int(input('Enter whether to look for MIN (-1) or MAX (1) voltage for {}'.format(erp_tochange)))

                # Change the temporary ERP time window parameters
                params.erp_windows[erp_tochange] = new_time_win

                try:
                    # Use new parameters to find either minimum or maximum value in that range
                    erp_peaks = self.find_minmax_erp(evoked, erp_peaks, erp_tochange, new_time_win, subject_name=subject, plot=plot, verbose = verbose, savefig=False, results_foldername=results_folder, exp_folder=exp_folder)
                except:
                    print('Something went wrong with manual ERP detection, try again.')

        for erp in params.erp_windows.keys():
            if erp not in erp_peaks:
                erp_peaks[erp] = [np.nan, np.nan]

        # Add new temporary ERP to the main dataframe
        df_erps_temp = self.erp_dict_to_df(erp_peaks, params.erp_windows, subject, session, group, event=event, channel=channel)

        # Combine with master dataframe
        master_erp_df = pd.concat([master_erp_df, df_erps_temp])
        print('ERPs have been found and added to the dataframe for {}'.format(subject))
        
        # Calculate relative peak-to-peak amplitudes between the ERPs
        self.calculate_relative_amplitudes(master_erp_df)
        
        return master_erp_df

    def group_erp_analysis(self, manage, analysis, params, trial_types, response_status, dataframe,  auto = True, plot = False, channels = None, evoked_file_extension = None, str_to_extract = None):
    
        if channels is None:
            raise ValueError("Must specify list of channels")
        
        for trial_type in trial_types:
            # Fetch BIDS data for the current trial type
            bids = manage.get_bids(extensions=[f'_{trial_type}{evoked_file_extension}'], str_to_extract=[str_to_extract], modalities=['eeg'])
            manage.add_classifier(bids, response_status, designation=[1,0])
            
            for entry in bids:
                subject = entry['subject']
                session = entry['session']
                task = entry['task']
                response = entry['group']

                for channel in channels:
                    evoked = mne.read_evokeds(entry['data'], condition=0)
                    evoked_subset = evoked.copy().pick(channel)
                    epochs = mne.read_epochs(f"{manage.root_folder}/{subject}/{session}/eeg/{subject}_{session}_{trial_type}_ar_clean_{task}-epo.fif")

                    temp_df = pd.DataFrame()
                    erp_data = analysis.erp_analysis(evoked_subset, epochs, params, auto = auto, verbose = False, master_erp_df=temp_df, plot=plot, subject=subject, session = session, group = response, event=trial_type, channel=channel, results_folder=None, exp_folder=None)

                dataframe = pd.concat([dataframe, erp_data])

        return dataframe, bids
    
    def calculate_psd(self, epochs, subjectname, fminmax=[1,50], method='welch', window='hamming',
                    window_duration=2, window_overlap=0.5, zero_padding=3, tminmax=[None, None],
                    verbose=False, plot=True):
        """
        Calculate power spectrum density with FFT/Welch's method and plot the result.

        Parameters
        ----------
        epochs: Epochs-type (MNE-Python) EEG file
        subjectname: A string for subject's name
        fminmax (optional): The minimum and maximum frequency range for estimating Welch's PSD
        window (optional): The window type for estimating Welch's PSD
        window_duration (optional): An integer for the length of that window
        window_overlap (optional): A float for the percentage of windows size 
                                    for overlap between the windows
        zero-padding (optional): A float for coefficient times window size for zero-pads
        tminmax (optional): A list of first and last timepoint of the epoch to include;
                            uses all epoch by default

        Returns
        -------
        psds: An array for power spectrum density values
        freqs: An array for the corresponding frequencies
        """

        # Calculate window size in samples and window size x coefs for overlap and zero-pad
        window_size = int(epochs.info['sfreq']*window_duration)
        n_overlap = int(window_size*window_overlap)
        n_zeropad = int(window_size*zero_padding)

        # N of samples from signal equals to windows size
        n_per_seg = window_size

        # N of samples for FFT equals N of samples + zero-padding samples
        n_fft = n_per_seg + n_zeropad

        # Calculate PSD with Welch's method
        spectrum = epochs.compute_psd(method=method, fmin=fminmax[0], fmax=fminmax[1], 
                                    n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap,
                                    window=window, tmin=tminmax[0], tmax=tminmax[1],
                                    verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)

        # Unit conversion from V^2/Hz to uV^2/Hz
        psds = psds*1e12

        # If true, print all the parameters involved in PSD calculation
        if verbose == True:
            print("---\nPSD ({}) calculation\n".format(method))
            print(spectrum)
            print('Time period:', str(tminmax))
            print('Window type:', window)
            print('Window size:', window_size)
            print('Overlap:', n_overlap)
            print('Zero-padding:', n_zeropad)
            print('\nSamples per segment:', n_per_seg)
            print('Samples for FFT:', n_fft)
            print('Frequency resolution:', freqs[1]-freqs[0], 'Hz')

        # If true, plot average PSD for all epochs and channels with channel PSDs
        if plot == True:
            plt.figure(figsize=(5,3), dpi=100)
            plt.plot(freqs,np.transpose(psds.mean(axis=(0))),color='black',alpha=0.1)
            plt.plot(freqs,psds.mean(axis=(0, 1)),color='blue',alpha=1)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (uV\u00b2/Hz)')
            plt.title("PSD,{} ({})".format(method, subjectname))
            plt.xlim(fminmax)
            plt.ylim(0,None)
            plt.grid(linewidth=0.2)
            plt.show()

        return [psds,freqs]

    def signal_quality_check(self, psds,freqs,band,b_name,subjectname,epochs):
        """
        Plot topographically PSD values and calculate median absolute deviation
        for the first and second half of the signal for signal reliability control.

        Parameters
        ----------
        psds: An array for power spectrum density values
        freqs: An array for corresponding frequencies
        band: A list of lower and higher frequency for the frequency band in interest
        b_name: A string for frequency band in interest
        subjectname: A string for subject's name
        epochs: Epochs-type (MNE-Python) EEG file

        Returns
        -------
        psd_max_mad_error: A float for max median absolute deviation for PSD values
        """
        # Divide the epochs into two
        idx_mid_epoch = round(len(psds[:,0,0])/2)
        psds_p1 = psds[:idx_mid_epoch,:,:]
        psds_p2 = psds[idx_mid_epoch:,:,:]

        # Define the bandpower indices
        low, high = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Get only the bandpower values that we need for both parts
        psd_band_ch_p1 = psds_p1[:,:,idx_band]
        psd_band_mean_epochs_p1 = psd_band_ch_p1.mean(axis=(2))
        psd_band_ch_p2 = psds_p2[:,:,idx_band]
        psd_band_mean_epochs_p2 = psd_band_ch_p2.mean(axis=(2))

        # Find Median Absolute Deviation along all the epochs for each channel
        psd_band_mean_epochs = np.concatenate((psd_band_mean_epochs_p1, psd_band_mean_epochs_p2))
        psd_median = np.median(psd_band_mean_epochs,axis=0)
        psd_mad = stats.median_abs_deviation(psd_band_mean_epochs,axis=0)

        # For each epoch in all channels, calculate the Z-score (MAD error) using median and MAD
        psd_mad_error_epochs_per_ch_p1 = np.zeros(shape=psd_band_mean_epochs_p1.shape)
        psd_mad_error_epochs_per_ch_p2 = np.zeros(shape=psd_band_mean_epochs_p2.shape)
        for i in range(len(psds[0,:,0])):
            psd_mad_error_epochs_per_ch_p1[:,i] = abs(psd_band_mean_epochs_p1[:,i] - psd_median[i])/psd_mad[i]
            psd_mad_error_epochs_per_ch_p2[:,i] = abs(psd_band_mean_epochs_p2[:,i] - psd_median[i])/psd_mad[i]

        # Average the Z-scores (MAD errors) for all epochs for the two parts -> get score for 19 channels
        psd_mad_error_per_ch_p1 = np.mean(psd_mad_error_epochs_per_ch_p1,axis=0)
        psd_mad_error_per_ch_p2 = np.mean(psd_mad_error_epochs_per_ch_p2,axis=0)

        # Find the average Z-score (MAD error) for the whole scalp (all channels together) for the two parts
        psd_mad_error_avg = [np.round(np.mean(psd_mad_error_per_ch_p1),3),
                            np.round(np.mean(psd_mad_error_per_ch_p2),3)]
        
        # Get the maximum of the two (for outputting it later)
        psd_max_mad_error = np.max(psd_mad_error_avg)

        # Find the average PSD values for each channel for both parts
        psds_all_channels_p1 = psds_p1.mean(axis=(0))
        psd_band_ch_p1 = psds_all_channels_p1[:,idx_band]
        psd_band_mean_ch_p1 = psd_band_ch_p1.mean(axis=(1))
        psds_all_channels_p2 = psds_p2.mean(axis=(0))
        psd_band_ch_p2 = psds_all_channels_p2[:,idx_band]
        psd_band_mean_ch_p2 = psd_band_ch_p2.mean(axis=(1))
        
        # Visually display the bandpower for both parts for visual inspection
        vmin = min([min(psd_band_mean_ch_p1),min(psd_band_mean_ch_p2)])
        vmax = max([max(psd_band_mean_ch_p1),max(psd_band_mean_ch_p2)])

        if psd_max_mad_error >= 2:
            sns.set_style("white",{'font.family': ['sans-serif']})
            fig,(ax1,ax2) = plt.subplots(ncols=2)
            fig.suptitle("MAD >= 2 ! Quality control for {} ({})".format(b_name,subjectname),y=1.1,x=0.575)
            im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p1,epochs.info,axes=ax1,vlim=[vmin,vmax],show=False)
            im,cm = mne.viz.plot_topomap(psd_band_mean_ch_p2,epochs.info,axes=ax2,vlim=[vmin,vmax],show=False)
            ax1.set_title("Epochs 0-{}\nAvg MAD error = {}".format(idx_mid_epoch-1,psd_mad_error_avg[0]))
            ax2.set_title("Epochs {}-{}\nAvg MAD error = {}".format(idx_mid_epoch,len(psds[:,0,0]),psd_mad_error_avg[1]))
            ax_x_start = 0.95
            ax_x_width = 0.04
            ax_y_start = 0.1
            ax_y_height = 0.9
            cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
            clb = fig.colorbar(im, cax=cbar_ax)
            clb.ax.set_ylabel('uV\u00b2/Hz'); 

        return psd_max_mad_error

    def bandpower_per_channel(self, psds, freqs, band, b_name, subjectname, epochs,
                          ln_normalization=False, verbose=True):
        """
        Find frequency band power in interest for all the channels.

        Parameters
        ----------
        psds: An array for power spectrum density values
        freqs: An array for corresponding frequencies
        band: A list of lower and higher frequency for the frequency band in interest
        b_name: A string for frequency band in interest
        subjectname: A string for subject's name
        epochs: Epochs-type (MNE-Python) EEG file

        Returns
        -------
        psd_band_mean_ch: An array for a frequency band power values for all the channels.
        """
        # Calculate the MAD error (z-score) of the bandpower to be sure of the quality
        _ = self.signal_quality_check(psds, freqs, band, b_name, subjectname, epochs)
        
        # Average all epochs together for each channels' PSD values
        psds_per_ch = psds.mean(axis=(0))

        # Pick only PSD values which are within the frequency band of interest
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        psds_band_per_ch = psds_per_ch[:,idx_band]

        # Average the PSD values within the band together to get bandpower for each channel
        bp_per_ch = psds_band_per_ch.mean(axis=(1))

        # If true, normalise the BP with natural logarithm transform
        if ln_normalization == True:
            bp_per_ch = np.log(bp_per_ch)
        
        if verbose == True:
            print('Finding bandpower within {} Hz with Ln normalisation set to {}'.format(
                band, str(ln_normalization)))

        return bp_per_ch    
    
    def psd_bands(self, bids, params, regions = False, plot=False, verbose=False):
        """
        Processes PSD bands and returns a DataFrame with the results.

        Parameters:
        - params: An object or dictionary containing the PSD options, including 'params' and 'bands'.
        - bids: A list of dictionaries containing information about subjects, tasks, and sessions.
        - epochs_file: The filename or path to the epochs data.

        Returns:
        - DataFrame containing the PSD band powers for all subjects.
        """
        df_psd_ch_bands = pd.DataFrame()

        for sub in bids:

            # Read epochs
            epochs = mne.read_epochs(fname=sub['data'], verbose=verbose)

            # Calculate Welch's power spectrum density
            psds, freqs = self.calculate_psd(
                epochs, 
                sub['subject'], 
                **params.psd_options['params'],
                verbose=verbose, 
                plot=plot
            )

            # Process each frequency band
            for band in params.psd_options['bands'].keys():
                # Compute band powers
                psd_ch_band_temp = self.bandpower_per_channel(
                    psds, 
                    freqs, 
                    params.psd_options['bands'][band], 
                    band,
                    sub['subject'], 
                    epochs
                )
                
                # Convert to DataFrame and add additional columns
                df_psd_ch_band_temp = manage.array_to_df(
                    sub['subject'], 
                    epochs, 
                    array_channels = psd_ch_band_temp
                )
                df_psd_ch_band_temp.insert(0, 'Condition', sub['task'])
                df_psd_ch_band_temp.insert(1, 'Timepoint', sub['session'])
                df_psd_ch_band_temp.insert(2, 'Band', band)

                # Concatenate with the main DataFrame
                df_psd_ch_bands = pd.concat([df_psd_ch_bands, df_psd_ch_band_temp])

        if regions:
            # Average the channels together for the specified regions
            df_psd_reg_bands = manage.df_channels_to_regions(df_psd_ch_bands.reset_index(), params.psd_options['brain_regions'])
            df_psd_reg_bands = df_psd_reg_bands.set_index(df_psd_ch_bands.index)
            df_psd_reg_bands.insert(0, 'Condition', df_psd_ch_bands['Condition'])
            df_psd_reg_bands.insert(1, 'Timepoint', df_psd_ch_bands['Timepoint'])
            df_psd_reg_bands.insert(2, 'Band', df_psd_ch_bands['Band'])
            
            return df_psd_reg_bands, df_psd_ch_bands
                
        else:
            return df_psd_ch_bands
        
    def alpha_reactivity(self, dataframe):
        """
        Calculate alpha reactivity from a list of DataFrames.

        Parameters:
        - dataframe_list: Dataframe containing PSD band power information.

        Returns:
        - Dictionary where keys are DataFrame identifiers (or indices) and values are DataFrames with calculated alpha reactivity.
        """
        # Choose only alpha band from the bandpowers dataframe
        df_psd_alpha = dataframe[dataframe['Band'] == 'Alpha']

        # Divide the alpha dataframe to EC and EO conditions
        df_psd_alpha_EC = df_psd_alpha[df_psd_alpha['Condition'] == 'EC'].reset_index(drop=True)
        df_psd_alpha_EO = df_psd_alpha[df_psd_alpha['Condition'] == 'EO'].reset_index(drop=True)

        # Merge the two conditions together to one dataframe by including only subjects who have both conditions
        df_psd_alpha_EC_EO = df_psd_alpha_EC.merge(df_psd_alpha_EO, on='Subject', how='inner', suffixes=('_EC', '_EO'))

        # Create new dataframe and calculate alpha reactivity for each subject
        alpha_reactivity = pd.DataFrame()
        alpha_reactivity['Subject'] = df_psd_alpha_EC_EO['Subject']
        alpha_reactivity['Timepoint'] = df_psd_alpha_EC_EO['Timepoint_EC']

        # Calculate alpha reactivity for each channel
        for ch in df_psd_alpha_EC.columns:
            if ch not in ['Subject', 'Condition', 'Band', 'Timepoint']:
                alpha_reactivity[ch] = (df_psd_alpha_EC_EO[f'{ch}_EC'] - df_psd_alpha_EC_EO[f'{ch}_EO']) / df_psd_alpha_EC_EO[f'{ch}_EC']
        
        return alpha_reactivity

    def compute_grand_averages(self, trial_types, sessions, channels, manage, response_status, evoked_file_extension= None):
        """
        Compute grand average evoked responses for different trial types, channels, and sessions.
        
        Parameters:
        - trial_types: List of trial types to process.
        - sessions: List of session identifiers.
        - channels: List of channels to consider.
        - manage: Object or module responsible for fetching and classifying BIDS data.
        - response_status: Classifier or status used for managing BIDS data.
        - evoked_file_extension: Extension of the file containing evoked data.
        
        Returns:
        - A dictionary with grand average evoked data and times for responders and non-responders.
        """
        if evoked_file_extension is None:
            raise ValueError('Must specify evoked file extension that is common across files you wish to average')
        
        # Initialize dictionaries for storing grand average data and times
        master_grand_evoked_data_resp = {
            session: {trial_type: {channel: None for channel in channels} for trial_type in trial_types}
            for session in sessions
        }

        master_grand_evoked_times_resp = {
            session: {trial_type: {channel: None for channel in channels} for trial_type in trial_types}
            for session in sessions
        }

        master_grand_evoked_data_nonresp = {
            session: {trial_type: {channel: None for channel in channels} for trial_type in trial_types}
            for session in sessions
        }

        master_grand_evoked_times_nonresp = {
            session: {trial_type: {channel: None for channel in channels} for trial_type in trial_types}
            for session in sessions
        }

        for trial_type in trial_types:
            # Fetch BIDS data for the current trial type
            bids = manage.get_bids(extensions=[f'{trial_type}{evoked_file_extension}'], str_to_extract=['AO'], modalities=['eeg'])
            manage.add_classifier(bids, response_status)
            
            for session in sessions:
                # Filter bids data by session
                session_data = [entry for entry in bids if entry['session'] == session]

                for channel in channels:
                    responder_evoked_signal = []
                    nonresponder_evoked_signal = []

                    for entry in session_data:
                        evoked = mne.read_evokeds(entry['data'], condition=0)
                        
                        if channel in evoked.ch_names:
                            evoked_subset = evoked.copy().pick(channel)
                        else:
                            # Skip if channel is not in the data
                            continue

                        if entry['group'] == 1:
                            responder_evoked_signal.append(evoked_subset)
                        else:
                            nonresponder_evoked_signal.append(evoked_subset)

                    if responder_evoked_signal:
                        grand_avg_resp = mne.grand_average(responder_evoked_signal)
                        master_grand_evoked_data_resp[session][trial_type][channel] = grand_avg_resp.data[0] * 1e6
                        master_grand_evoked_times_resp[session][trial_type][channel] = grand_avg_resp.times * 1e3
                    else:
                        master_grand_evoked_data_resp[session][trial_type][channel] = None
                        master_grand_evoked_times_resp[session][trial_type][channel] = None

                    if nonresponder_evoked_signal:
                        grand_avg_nonresp = mne.grand_average(nonresponder_evoked_signal)
                        master_grand_evoked_data_nonresp[session][trial_type][channel] = grand_avg_nonresp.data[0] * 1e6
                        master_grand_evoked_times_nonresp[session][trial_type][channel] = grand_avg_nonresp.times * 1e3
                    else:
                        master_grand_evoked_data_nonresp[session][trial_type][channel] = None
                        master_grand_evoked_times_nonresp[session][trial_type][channel] = None

        return {
            'master_grand_evoked_data_resp': master_grand_evoked_data_resp,
            'master_grand_evoked_times_resp': master_grand_evoked_times_resp,
            'master_grand_evoked_data_nonresp': master_grand_evoked_data_nonresp,
            'master_grand_evoked_times_nonresp': master_grand_evoked_times_nonresp
        }
    
    def calculate_complexity(self, manage, trial_types, response_status, channels = None, epoch_file_extension = None, str_to_extract = None):
    
        if channels is None:
            raise ValueError("Must specify list of channels")
        
        # Initialise dataframes
        df = pd.DataFrame()
        df_exp = pd.DataFrame()
        df_exp_channels = pd.DataFrame()

        for trial_type in trial_types:
            # Fetch BIDS data for the current trial type
            bids = manage.get_bids(extensions=[f'_{trial_type}{epoch_file_extension}'], str_to_extract=[str_to_extract], modalities=['eeg'])
            manage.add_classifier(bids, response_status, designation=[1,0])

            for i, entry in enumerate(bids):

                subject = entry['subject']
                session = entry['session']
                task = entry['task']
                response = entry['group']

                # Update df_exp and df_exp_channels with subject information
                df_exp.loc[i, 'Subject'] = subject
                df_exp.loc[i, 'Timepoint'] = session
                df_exp.loc[i, 'Task'] = task
                df_exp.loc[i, 'Trial_Type'] = trial_type
                df_exp.loc[i, 'Responder'] = response
                
                df_exp_channels.loc[i, 'Subject'] = subject
                df_exp_channels.loc[i, 'Timepoint'] = session
                df_exp_channels.loc[i, 'Task'] = task
                df_exp_channels.loc[i, 'Trial_Type'] = trial_type
                df_exp_channels.loc[i, 'Responder'] = response
                
                epochs = mne.read_epochs(fname=entry['data'], verbose=False)
                
                # Convert data file to dataframe 
                df_epochs = epochs.to_data_frame()
                
                # Calculate Lempel-Ziv Complexity
                sampen_values = []
                # fractal_nld = []
                for ch in channels:
                    sampen_ch = []
                    # fractal_nld_ch = []
                    for epo in df_epochs['epoch'].unique():
                        epo_signal = df_epochs[df_epochs['epoch'] == epo][ch]

                        # Calculate sample and fractal entropy
                        sampen, _ = nk.entropy_sample(epo_signal.to_numpy(), delay=1, dimension=2)
                        # fd, _ = nk.fractal_nld(epo_signal, corrected=False)

                        sampen_ch.append(sampen)
                        # fractal_nld_ch.append(fd)

                    # Add channel values dataframe
                    df_exp_channels.loc[i, ch] = np.mean(sampen_ch)
                    
                    # Store average values
                    sampen_values.append(np.mean(sampen_ch))
                
                # Average all the channels' LZC values to get a single value for the subject & add to master dataframe
                sampen_mean = np.mean(sampen_values)
                df_exp.loc[i, 'sampen'] = sampen_mean

        # Add the current timepoint data to the master dataframe
        df = pd.concat([df, df_exp])
        
        return df, df_exp_channels