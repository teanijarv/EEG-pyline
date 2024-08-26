class PylineParameters:
    def __init__(self, reference = None, eog_channels=None, stimulus_channel=None, filter_design=None, line_noise =  None, ica = None,
                 epoch_tminmax=None, baseline_correction=None, event_dict=None, button_id=None, rejection_criteria=None, condition_comp = None,
                 erp_windows = None, channel_picks = None):
        """
        Initialize EEG Parameters.

        Parameters:
        - eog_channels (list of str): List of EOG channel names.
        - stimulus_channel (str): Name of the stimulus channel.
        - filter_design (dict): Dictionary containing filter design parameters.
        - epoch_tminmax (list of float): Epoch time window from event/stimuli.
        - baseline_correction (tuple or None): Baseline correction time window.
        - event_dict (dict): Dictionary containing event names with IDs.
        - button_id (int): Button press ID.
        """
        self.reference = reference if reference is not None else 'average'
        self.projector = True if self.reference == 'average' else False
        self.eog_channels = eog_channels if eog_channels is not None else ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
        self.stimulus_channel = stimulus_channel if stimulus_channel is not None else 'Status'
        self.filter_design = filter_design if filter_design is not None else {
                'l_freq': 0.4, 'h_freq': 30, 'filter_length': 'auto', 
                'l_trans_bandwidth': 'auto', 'h_trans_bandwidth': 'auto', 'n_jobs': None,
                'method': 'fir', 'iir_params': None, 'phase': 'zero',
                'fir_window': 'hamming', 'fir_design': 'firwin', 'pad': 'reflect_limited'
            }
        self.line_noise = line_noise if line_noise else None
        self.ica = ica if ica is not None else {
            'use': False,
            'n_components': 15,
            'method': 'fastica',
            'plot': False,
            'autoselect': False
        }
        self.epoch_tminmax = epoch_tminmax if epoch_tminmax is not None else [-0.2, 0.8]
        self.baseline_correction = baseline_correction if baseline_correction is not None else None
        self.event_dict = event_dict if event_dict is not None else {
            'AXCPT': {'Filler/4': 4, 'Filler/8': 8, 'AX': 2, '*X': 6
            , 'button_id': 128}, 
            'AO': {'target after 1 standard': 17, 'target after 3 standards': 19,
              'target after 5 standards': 21, 'target after 7 standards': 23,
              'target after 9 standards': 25, 'target after 11 standards': 27, 'button_id': 32},
            'resting': {'Initiated': 1, 'Start Tone': 4, 'End Tone': 2}
        }
        self.button_id = button_id if button_id is not None else 128
        self.rejection_criteria = rejection_criteria if rejection_criteria is not None else {'eeg': 0.00015} # 150 µV
        self.condition_comp = condition_comp if condition_comp is not None else ["AX", "*X"]

        self.erp_windows = erp_windows if erp_windows is not None else {'N1' : [40, 170, -1],
        'N2' : [180, 350, -1],'P2' : [100, 260, 1],'P3' : [270, 500, 1]}
        self.channel_picks = channel_picks if channel_picks is not None else ['Fz', 'Cz', 'Pz']
        self.picks = None

        self.psd_options = {
        'bands': {
            'Delta': [1, 3.9],
            'Theta': [4, 7.9],
            'Alpha': [8, 12],
            'Beta': [12.1, 30]
        },
        'brain_regions': {
            'Left frontal': ['AF3', 'F3', 'FC1'],
            'Right frontal': ['AF4', 'F4', 'FC2'],
            'Left temporal': ['F7', 'FC5', 'T7'],
            'Right temporal': ['F8', 'FC6', 'T8'],
            'Left posterior': ['CP5', 'P3', 'P7'],
            'Right posterior': ['CP6', 'P4', 'P8']
        },
        'params': {
            'method': 'welch',
            'fminmax': [1, 30],
            'window': 'hamming',
            'window_duration': 2.5,
            'window_overlap': 0.5,
            'zero_padding': 3
        }}