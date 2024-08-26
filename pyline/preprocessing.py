# Import packages
import mne, os, warnings
import pandas as pd
import numpy as np
from autoreject import (get_rejection_threshold, AutoReject)
from mne.preprocessing import ICA

from . import PylineData

class PylinePreprocessing:
    def __init__(self):
        """
        Initialize the EEGPreprocessing object.

        Parameters:
        - 
        """

    """
    Transformations of the data.
    
        Functions:
        - resample ()
        - filter ()
        - csd ()
        - pyline ()
    
    """
    def ssp (self, data, params, verbose = False): 
            
        if verbose==True: print('---\nAPPLYING SSP FOR EOG-REMOVAL\n')

        eog_projs, _ = mne.preprocessing.compute_proj_eog(data.raw,n_grad=0,n_mag=0,n_eeg=1,reject=None,
                                                        no_proj=True,ch_name=params.eog_channels,verbose=verbose)
        data.raw.add_proj(eog_projs,remove_existing=False)
        data.raw.apply_proj()
        if params.reference == 'mastoids':
            # Assuming params.eog_channels is a list of channel names
            eog_channels = [x for x in params.eog_channels if x not in ['EXG1', 'EXG2']]
            data.raw.drop_channels(eog_channels)
        else:
            data.raw.drop_channels(params.eog_channels)

    def ica (self, data, n_components = 32, method = 'fastica', plot = False, autoselect = False):
        """ """
        
        filt_raw = data.raw.copy().filter(l_freq=1.0, h_freq=30)

        ica = ICA(n_components=n_components, method= method, random_state=23)
        
        ica.fit(filt_raw)
        
        if plot:
            ica.plot_components()
            ica.plot_sources(data.raw, show_scrollbars=False)
        
        if autoselect: 
            eog_indices = ica.find_bads_eog(data.raw)
            ica.exclude = eog_indices[0]
            ica.apply(data.raw)
        
        data.ica = ica
        return data
    
    def resample(self, data, events, sfreq = None):
        """
        Resample raw EEG data.

        Parameters:
        - raw_data (mne.io.Raw): Raw EEG data to filter.

        Returns:
        - filtered_data (mne.io.Raw): Filtered raw EEG data.
        """
        # if data.raw is None:  
        #     raise ValueError("No raw data file in data object")
        
        if sfreq is None or not isinstance(sfreq, int):  
            raise ValueError("Sampling frequency must be specified as integer")

        # if data.ica is not None:
        #     print("Resampling ica data file")
        resampled_data, updated_events = data.copy().resample(sfreq=sfreq, events = events)
        # else:
        #     print("Resampling raw data file")
        #     resampled_data, updated_events = data.raw.copy().resample(sfreq=sfreq, events = events)
            
        print("Updating data object") 
        # data.resampled = resampled_data
        data.events = updated_events
        return resampled_data
    
    def filter(self, data, params, plot= False, savefig=False, verbose = False):
        """
        Filter raw EEG data.

        Parameters:
        - raw_data (mne.io.Raw): Raw EEG data to filter.
        - filter_design: A dictionary of all the filter parameters (see MNE raw.filter or create_filter functions)
        - line_remove (optional): A boolean whether to remove power-line noise (50Hz) with a Notch filter or not
        - eog_channels (optional): A boolean whether to remove EOG noise or not, requires list of EOG channels
        - plot_filter (optional): A boolean whether to plot the band-pass filter
        - savefig (optional): A boolean whether to save the filter design

        Returns:
        - filtered_data (mne.io.Raw): Filtered raw EEG data.
        """
        #filtered_data =  data.filter(self.filter_design)
        if data is None:  
            raise ValueError("No data file saved in data object passed")
        
        # if data.filtered is not None:  
        #     raise ValueError("Filtered data already exists in data object. Pass non-filtered data.")
        
        # if data.ica is None: 
        print('---\nAPPLYING FILTER\n')
        filtered_data = data.copy().filter(**params.filter_design, verbose=verbose)
        print("Updating data object")
        
        if plot == True: # Review and fix where data.raw and data.resampled are passed
            temp_params = params.filter_design
            temp_params.pop('n_jobs')
            temp_params.pop('pad')
            filter_params = mne.filter.create_filter(data.raw.get_data(),data.resampled.info['sfreq'],**temp_params)
            
            freq_ideal = [0,params.filter_design['l_freq'],params.filter_design['l_freq'],
                        params.filter_design['h_freq'],params.filter_design['h_freq'],data.resampled.info['sfreq']/2]
            gain_ideal = [0, 0, 1, 1, 0, 0]

            fig, axs = plt.subplots(nrows=3,figsize=(8,8),layout='tight',dpi=100)
            mne.viz.misc.plot_filter(filter_params,data.resampled.info['sfreq'],freq=freq_ideal,gain=gain_ideal,
                                    fscale='log',flim=(0.01, 80),dlim=(0,6),axes=axs,show=False)
            if savefig == True:
                plt.savefig(fname='Data/filter_design.png',dpi=300)
            plt.show()

        if params.line_noise != None:
            if verbose==True: print('---\nAPPLYING NOTCH FILTER\n')
            filtered_data = filtered_data.filtered.notch_filter([params.line_noise])

            # print("Filtering resampled data file")
            # filtered_data = prep.filter_raw_data(data.resampled, eeg_params.filter_design, line_remove=None, eog_channels= eeg_params.eog_channels if ssp_projection else None,
            # plot_filt=plot, savefig=savefig, verbose = verbose)
        #else:
        #     print("Filtering resampled data file")
        #     filtered_data = prep.filter_raw_data(data.ica, eeg_params.filter_design, line_remove=None, eog_channels= eeg_params.eog_channels if ssp_projection else None, 
		# 	plot_filt=plot, savefig=savefig, verbose = verbose)

        print("Updating data object")    
        return filtered_data
    
    def csd(self, data): # update description
        """
        Find events in raw EEG data.

        Parameters:
        - raw_data (mne.io.Raw): Raw EEG data.

        Returns:
        - events (ndarray): Array containing event onsets.
        """
        # if data.raw is None:  
        #     raise ValueError("No raw data file in data object")
        
        # if data.csd is not None:  
        #     raise ValueError("CSD data already exists in data object.")
        
        # if data.resampled is None and data.filtered is None:  
        #     print("Computing CSD for raw data file")
        #     csd_data = mne.preprocessing.compute_current_source_density(data.raw)
        # elif data.resampled is None:
        #     print("Computing CSD for filtered data file")
        #     csd_data = mne.preprocessing.compute_current_source_density(data.filtered)
        # else:
        #     print("Computing CSD for resampled data file")
        csd_data = mne.preprocessing.compute_current_source_density(data)

        print("Updating data object") 
        # data.csd = csd_data
        return csd_data
    
    def crop(self, data, params):
        """
        Crop the EEG signal based on event markers.

        Parameters:
        raw : mne.io.Raw
            The raw EEG data.
        stimulus_channel : str
            The name of the stimulus channel used to find events.
        subject_name : str
            The name of the subject for warning messages.

        Returns:
        cropped_raw : mne.io.Raw
            The cropped EEG data.
        """

        # Initialize variables
        tminmax = None
        
        # Process events and determine cropping boundaries
        if len(data.events) >= 3:
            tminmax = [data.events[0][0] / data.raw.info['sfreq'], data.events[-1][0] / data.raw.info['sfreq']]
            # Warn if there are more than 3 events
            if len(data.events) > 3:
                warnings.warn('\nMore than 3 event points found for {}\n'.format(data.filename))
        elif len(data.events) == 1 or len(data.events) == 2:
            warnings.warn('\nOnly 1 or 2 event point(s) found for {}\n'.format(data.filename))
            
            if data.events[0][0] > 100000:
                tminmax = [0, data.events[0][0] / data.raw.info['sfreq']]
            else:
                tminmax = [data.events[0][0] / data.raw.info['sfreq'], None]
        else:
            warnings.warn('\nNO event points found for {}\n'.format(data.filename))
        
        # Crop the raw data based on the event markers
        if tminmax is not None:
            cropped_raw = data.raw.crop(tmin=tminmax[0], tmax=tminmax[1])
            print(('Event markers are following:\n{}\nStarting point: {} s\nEnding point: {} s\n'
                'Total duration: {} s').format(data.events, tminmax[0], tminmax[1], tminmax[1] - tminmax[0]))
            
            # Warn if the signal length is not within the expected range
            if not (230 <= (tminmax[1] - tminmax[0]) <= 250):
                warnings.warn('\nRaw signal length is not between 230-250s for {}\n'.format(data.filename))
        else:
            print('Signal NOT cropped.')

        # Drop the stimulus channel
        cropped_raw = cropped_raw.drop_channels(params.stimulus_channel)
        
        return cropped_raw

    def pyline(self, data, params, manage, analysis, tasktype = None, ssp_projection = True, sfreq=None, plot=False, savefig=False, verbose=False):
        """
        Preprocess raw EEG data using automatic settings.

        Parameters:
        - data (object): The data object containing raw EEG data and events.
        - params (object): Parameters object containing EEG parameters and ICA settings.
        - ssp_projection (bool): Whether to apply SSP projection.
        - sfreq (int, optional): Sampling frequency.
        - plot (bool): Whether to plot the preprocessed data.
        - savefig (bool): Whether to save the plots.
        - verbose (bool): Whether to print verbose output.
        """
        # Validate input types
        if not isinstance(params, object):
            raise TypeError("params must be an object")
        if not isinstance(params.ica, dict):
            raise TypeError("params.ica must be a dictionary")
        if tasktype is None :
            raise TypeError("You must specify either resting or task when running pyline")
        if not isinstance(ssp_projection, bool):
            raise TypeError("ssp_projection must be a boolean")
        if sfreq is not None and not isinstance(sfreq, int):
            raise TypeError("sfreq must be an integer or none")
        if not isinstance(plot, bool):
            raise TypeError("plot must be a boolean")
        if not isinstance(savefig, bool):
            raise TypeError("savefig must be a boolean")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        # Dictionary to map condition types to methods
        condition_specific_code = {
            "resting": self._pyline_resting,
            "task": self._pyline_task,
            # Add more conditions as needed
        }

        # Verify that the tasktype is valid and call the appropriate method
        if tasktype not in condition_specific_code:
            raise ValueError(f"Unsupported condition type: {tasktype}. Add separate function to object EEGprocessing code.")

        # Call the condition-specific method with all relevant parameters
        return condition_specific_code[tasktype](data, params, manage, analysis, tasktype, ssp_projection, sfreq, plot, savefig, verbose)

    def _pyline_resting(self, data, params, manage, analysis, tasktype, ssp_projection, sfreq, plot, savefig, verbose):
        
        for entry in data:

            # Set filenames, output directory, and create data object passing params, filename and task identifiers.
            filename = f"{entry['subject']}_{entry['session']}"
            print(f"Running for {filename}")

            output_path = manage.clean_folder / entry['subject'] /entry['session'] / entry['modality']
            data = PylineData(manage.load_eeg(entry['data'], params), filename, tasktype=tasktype)
            paradigm = entry['task']
            params.picks = mne.pick_types(data.raw.info, eeg=True, stim=False) # add picks to params

            # Find events, crop data, apply either ICA or SSP to remove major (repetitive) artifacts, then resample and/or filter
            if params.ica['use']:
                self.events_finder(data, params)

                data.raw = self.crop(data, params)

                ica_kwargs = {k: v for k, v in params.ica.items() if k != 'use'}
                self.ica(data, **ica_kwargs) # Automated ICA rejection applied in arguments

            elif ssp_projection:
                self.events_finder(data, params)

                data.raw = self.crop(data, params)

                self.ssp(data, params, verbose=verbose)

            if sfreq == None:
                data.filtered = self.filter(data.raw, params, plot, savefig, verbose)

            else:
                data.resampled = self.resample(data.raw, data.events, sfreq=sfreq)

                data.filtered = self.filter(data.resampled, params, plot, savefig, verbose)
                    
            epochs = self.create_epochs(data.filtered, params, tasktype='resting', epo_duration=5)

            ar_epochs = self.reject_auto(epochs)

            analysis.export(ar_epochs, type='fif', outputdir=output_path, filename=f"{data.filename}_{paradigm}")

        return

    def _pyline_task(self, data, params, manage, analysis, tasktype, ssp_projection, plot, savefig, verbose):
        if data[0]['task'] == 'AXCPT':
            processing_data = pd.DataFrame(columns=['Subject', 'Group', 'Timepoint', 'Task', 'Time', "AX-cue", "AX-target", "AB-cue", "AB-target", "BX-cue", "BX-target", "AX-cue_ar", "AX-target_ar", "AB-cue_ar", "AB-target_ar", "BX-cue_ar", "BX-target_ar"]) 
        elif data[0]['task'] == 'AO':
            processing_data = pd.DataFrame(columns=['Subject', 'Group', 'Timepoint', 'Task', 'Time', "Standard", "Target", "Standard_ar", "Target_ar"]) 

        trial_outcomes = pd.DataFrame()

        for entry in data:

            # Set filenames, output directory, and create data object passing params, filename and task identifiers.
            filename = f"{entry['subject']}_{entry['session']}"
            print(f"Running for {filename}")

            output_path = manage.clean_folder / entry['subject'] /entry['session'] / entry['modality']
            data = PylineData(manage.load_eeg(entry['data'], params), filename, tasktype=tasktype)
            paradigm = entry['task']
            params.picks = mne.pick_types(data.raw.info, eeg=True, stim=False) # add picks to params

            # Code specific to task data
            if params.ica['use']:
                self.events_finder(data, params)

                ica_kwargs = {k: v for k, v in params.ica.items() if k != 'use'}
                self.ica(data, **ica_kwargs) # Automated ICA rejection applied in arguments
            
            elif ssp_projection:
                self.events_finder(data, params)
                self.ssp(data, params, verbose=verbose)
            
            data.filtered = self.filter(data.raw, params, plot, savefig, verbose)

            # Split the events into the respective categories for each task
            AX_cue, AX_target, AB_cue, AB_target, BX_cue, BX_target, background_AX, background_A = self.split_events(data, params, task="AXCPT")

            # Create event dictionary for creating epochs (you can include activity preceding an AX or A if desired)
            event_dict = {
            "AX-cue": [AX_cue, {'AX-cue': 4}],
            "AB-cue": [AB_cue,{'A*-cue': 4}],
            "BX-cue": [BX_cue, {'BX-cue': 8}],
            "AX-target": [AX_target,{'AX-target': 2}],
            "AB-target": [AB_target, {'A*-target': 8}],
            "BX-target": [BX_target,{'BX-target': 6}]
            }
            
            epoch_dict = {} # Initialize an empty dictionary to store epoch variables
            # Epoch around events of interest for each task, passing the data.filtered 

            for key, value in event_dict.items(): # Iterate over the dictionary items and create epochs
                epochs = self.create_epochs(data.filtered, params, tasktype='task', 
                                                events = value[0], event_id = value[1],
                                                title=f"{key.capitalize()} Epochs (GFP without AR)",
                                                epo_duration=None,
                                                plot=False)
                epoch_dict[key] = epochs

                # Add number of cleaned epochs to processing dataframe 
                processing_data.loc[entry, key] = len(epochs)

            # # Process epochs with autoreject, define dictionary of processed epochs for creating evoked objects
            ar_dict = {}

            for key, value in epoch_dict.items():
                # Perform autoreject on each epoch object in dictionary
                ar = self.reject_auto(value, plot=False)

                # Update dictionary with processed epochs
                ar_dict[f"{key}_ar"] = ar

            # Determine the lowest number of epochs across conditions so epochs can be made the same. This makes the SNR comparable across trials.
            epoch_lens = []

            for key in ar_dict.keys():
                epoch_lens.append(len(ar_dict[key].events))

            minimum_epochs = min(epoch_lens)

            # Use minimum to randomly sample epochs. Export .fif and numpy array for analysis.alpha_reactivity
            for key, value in ar_dict.items():
                if len(value) > minimum_epochs:
                    ar_dict[key] = self.select_random_epochs(value, minimum_epochs) 

                analysis.export(ar_dict[key], type='fif', outputdir=output_path, filename=f"{data.filename}_{key}_clean_{paradigm}")

            # # Add number of cleaned epochs to processing dataframe for each participant 
            processing_data.loc[entry, 'min_epochs_post_ar'] = minimum_epochs

            # Create evoked objects for each trial type. Export for later ERP analysis.
            for key, value in ar_dict.items():
                evoked = analysis.create_evoked(value)
                analysis.export(evoked, type='fif', outputdir=output_path, filename=f"{data.filename}_{key}_evoked_{paradigm}")

            # Now, extract relevant processing and task-related data.
            trial_outcomes= analysis.extract_trial_data(trial_outcomes, data, data, entry, params, task = 'AXCPT')
            
        analysis.export(trial_outcomes, type = 'csv', outputdir=manage.analysis_folder, filename=f'{paradigm}_psychomotor')
        analysis.export(processing_data, type = 'csv', outputdir=manage.analysis_folder, filename=f'{paradigm}_processing_epoch_data') 

        return

    def events_finder(self, data, params, plot = False, task = None):
        """
        Find events in raw EEG data.

        Parameters:
        - raw_data (mne.io.Raw): Raw EEG data.

        Returns:
        - events (ndarray): Array containing event onsets.
        """
        events = mne.find_events(data.raw, stim_channel=params.stimulus_channel, consecutive=False, output='onset')
        data.events = events

        if plot:
            if task is None:
                raise ValueError('tasktype must be specified to plot events')
            fig = mne.viz.plot_events(
            events, event_id=params.event_dict[task], sfreq=data.raw.info["sfreq"], first_samp=data.raw.first_samp
            )
            return events, fig
        else:
            return events
        
    def split_events(self, data, params, task=None):
        if task is None:
            raise ValueError("Task type must be specified")

        if data.events is None:
            raise ValueError("Events have not been found. Run 'events_finder' first.")

        task_specific_code = {
            "AXCPT": self.split_events_AXCPT,
            "AO": self.split_events_AO
            # add more tasks as needed,
            # "another_task": self._split_events_another_task,
        }

        if task not in task_specific_code:
            raise ValueError(f"Unsupported task type: {task}. Add separate function to object EEGprocessing code.")
        
        return task_specific_code[task](data, params)
    
    def split_events_AXCPT(self, data, params):

        AX_cue = [] # AX-cue
        AX_target = [] # AX-target
        AB_cue = [] # A*-cue
        AB_target = [] # A*-target
        BX_cue = [] # BX-cue
        BX_target = [] # BX-target
        background_AX = [] # B preceding AX presentation
        background_A = [] # B preceding A*presentation

        for m in range(len(data.events) - 2):
            cue, target, next_event = data.events[m:m+3]
            cue_type, target_type, next_type = cue[2], target[2], next_event[2]

            # A-X successful responses
            if (
                cue_type == params.event_dict['AXCPT']['Filler/4']
                and target_type == params.event_dict['AXCPT']['AX']
                and next_type == params.button_id # only include those AX combinations with a button press
            ):
                AX_cue.append(cue)
                AX_target.append(target)
                
            # A-* successful responses
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/4']
                and target_type == params.event_dict['AXCPT']['Filler/8']
                and next_type != params.button_id # only include those A* combinations with no button press
            ):
                AB_cue.append(cue)
                AB_target.append(target)

            # *-X successful responses
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['*X']
                and next_type != params.button_id # only include those *-X combinations with no button press
            ):
                BX_cue.append(cue)
                BX_target.append(target)
                
            # B-AX (B's preceding AX cue-target presentations)
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['Filler/4']
                and next_type == params.event_dict['AXCPT']['AX'] 
            ):
                background_AX.append(cue)

            # B-AB (B's preceding AB cue-target presentations)
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['Filler/4']
                and next_type == params.event_dict['AXCPT']['Filler/8'] 
            ):
                background_A.append(cue)
            

        AX_cue = np.asarray(AX_cue)
        AX_target = np.asarray(AX_target)
        AB_cue = np.asarray(AB_cue)
        AB_target = np.asarray(AB_target)
        BX_cue = np.asarray(BX_cue)
        BX_target = np.asarray(BX_target)
        background_AX = np.asarray(background_AX)
        background_A = np.asarray(background_A)

        return (
              AX_cue, AX_target, AB_cue, AB_target, BX_cue, BX_target, background_AX, background_A
        )
    
    def split_events_AO(self, data, params):

        # Create an array of target tone events which have been responded with a button press
        standard = []
        one_standard= []
        three_standards = []
        five_standards = []
        seven_standards = []
        nine_standards= []
        eleven_standards= []

        responses = [k for k in params.event_dict['AO'].values()]
        responses.remove(32)

        # Iterate through events
        for m in range(len(data.events) - 1):
            cue, response = data.events[m:m+2]
            cue_type, response_type = cue[2], response[2]

            # Check for one and three standard tones preceding target tone
            if cue_type == 17 and response_type == 32:
                one_standard.append(cue)
            elif cue_type == 19 and response_type == params.event_dict['AO']['button_id']:
                three_standards.append(cue)
            elif cue_type == 21 and response_type == params.event_dict['AO']['button_id']:
                five_standards.append(cue)
            elif cue_type == 23 and response_type == params.event_dict['AO']['button_id']:
                seven_standards.append(cue)
            elif cue_type == 25 and response_type == params.event_dict['AO']['button_id']:
                nine_standards.append(cue)
            elif cue_type == 27 and response_type == params.event_dict['AO']['button_id']:
                eleven_standards.append(cue)
            elif cue_type == 2: # and response_type in responses:
                standard.append(cue)

        return (
            standard, 
            one_standard,
            three_standards,
            five_standards,
            seven_standards,
            nine_standards, 
            eleven_standards
        )
        
    def create_epochs(self, data, params, tasktype = None, events = None, event_id = None, title = None, epo_duration = None, plot=False):

        if tasktype is None:
                raise ValueError("Must specify either 'resting' or 'task.")
    
        if tasktype == 'resting':
            if epo_duration is None:
                raise ValueError("Duration must be specified for epoching.")
            
            epochs = mne.make_fixed_length_epochs(data, duration=epo_duration, preload=True)

        elif tasktype == 'task':
            if events is None or event_id is None or title is None:
                 raise ValueError("Events, event_id, and title must all be specified for epoching.")
            
            epochs = mne.Epochs(data, events, event_id, tmin=params.epoch_tminmax[0],
                                tmax=params.epoch_tminmax[1], baseline=params.baseline_correction,
                                picks=params.picks, preload=True)
        if plot:
            fig = epochs.plot_image(title=title)
            return epochs, fig
        else:
            return epochs
        
    def reject_auto (self, epochs, method = 'random_search', plot = False):
        if not isinstance(epochs, (mne.epochs.Epochs, mne.epochs.EpochsArray)):
            raise ValueError ("Reject auto function requires epochs. Run create_epochs.")
        reject_criteria = get_rejection_threshold(epochs)
        print('Dropping epochs with rejection threshold:', reject_criteria)
        
        temp_epochs = epochs.copy()
        temp_epochs.drop_bad(reject=reject_criteria)

        ar = AutoReject(thresh_method=method, random_state=1)
        ar.fit(epochs)
        clean_epochs, reject_log = ar.transform(temp_epochs, return_log=True)
        
        if plot:
            reject_log.plot('horizontal')
            clean_epochs.plot_image(title="GFP with AR ")
            
        return clean_epochs
    
    def select_random_epochs(self, epochs, num_epochs_to_select):
        """
        Selects a random group of epochs from an MNE Epochs object.
        
        Parameters:
        - epochs (mne.Epochs): The MNE Epochs object containing all epochs.
        - num_epochs_to_select (int): Number of epochs to randomly select.
        
        Returns:
        - random_epochs (mne.Epochs): The subset of epochs randomly selected.
        """
        # Get the total number of epochs in the Epochs object
        total_epochs = len(epochs)

        if num_epochs_to_select > total_epochs:
            raise ValueError(f'Number of epochs to sub-sample has to be equal to or less than {total_epochs}')
        
        # Generate random indices to select epochs
        random_indices = np.random.choice(total_epochs, size=num_epochs_to_select, replace=False)
        
        # Select the epochs using the random indices
        random_epochs = epochs[random_indices]
    
        return random_epochs