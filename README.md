# EEG-pyline: EEG pipeline in Python
[![DOI](https://zenodo.org/badge/495300654.svg)](https://zenodo.org/badge/latestdoi/495300654)

This is an EEG pipeline for resting and task EEG pre-processing and analyses used at the UniSC's Thompson Institute created by Toomas Erik Anijärv.

The main aim for creating this pipeline was to make EEG analysis in Python easier for other researchers who are not too familiar with programming but also do not want to use other commercial blackbox-style software. Using ready-made Jupyter notebooks, it is easy to get started with EEG data pre-processing, spectral analysis, and ERP analysis. 

Different studies that have been using this pipeline for their EEG data analysis steps can be found as study notebooks. They include also longer explanation on why some parameters or methods have been chosen for pre-processing and/or analysis parts.

## Notebook templates
Find in folder `/templates`. Different notebooks for using as templates for your EEG data analysis needs.

### Pre-processing
`/resting_state/resting_preprocessing.ipynb` - importing raw resting state EEG (.bdf) files, re-referencing, applying bandpass (e.g., 1-30 Hz) FIR filter, cropping the signal, removing EOG noise with SSP, dividing signal into equal-sized epochs (e.g., 5 seconds), performing artefact rejection with Autoreject algorithm, exporting the cleaned EEG signals (.fif).

`/auditory_oddball/oddball_erp_preprocessing.ipynb` - importing raw auditory oddball task EEG (.bdf) files, re-referencing, applying bandpass (e.g., 1-15 Hz) FIR filter, removing EOG noise with SSP, finding event points and including only the target tone followed by participant's button press, dividing the signal into 1-second epochs (i.e., -0.2s - 0s is pre-stimulus, 0s is stimulus or target tone, 0 - 0.8s is post-stimulus), performing artefact rejection with Autoreject algorithm, exporting the cleaned EEG signal (.fif).

`/sart/sart_erp_preprocessing.ipynb` - importing raw sustained attention response task (SART) task EEG (.bdf) files, re-referencing, applying bandpass (e.g., 1-30 Hz) FIR filter, removing EOG noise with SSP, finding event points and including only correct GO tasks (which were followed by participant's button press) and correct NO-GO tasks (which weren't followed by a button press) and exporting that, dividing the signal into 1.2-second epochs (i.e., -0.2s - 0s is pre-stimulus, 0s is the event, 0 - 1s is post-stimulus), performing artefact rejection with Autoreject algorithm, exporting the cleaned EEG signal (.fif).


### Spectral analysis
`/resting_state/resting_classic_bp_analysis.ipynb` - importing clean resting state EEG (.fif) files, calculating Welch's power spectrum density (PSD) for each region of interest (e.g., averaging channels together or channel-by-channel) and divides the PSD estimates to pre-defined frequency bands (e.g., delta, theta, alpha, beta). Optionally it is possible to calculate the (alpha) asymmetry using channels from each hemispheres and also find relative band powers by dividing the absolute band power value with broadband power. Finally, the results are exported as Excel spreadsheets for each band power, region and participant.

`/resting_state/resting_aperiodicfit_bp_analysis.ipynb` - importing clean resting state EEG (.fif) files, calculating Welch's PSD for each region of interest, estimate the aperiodic 1/f-like component of the spectra with specparam (FOOOF) algorithm, and export the aperiodic component's parameters (i.e., exponent and offset). Remove the aperiodic component of the spectra (i.e., flatten the spectra) and use pre-defined band of interest (e.g., alpha) to find its peak parameters (i.e., center frequency and peak width) and absolute and relative PSDs. Finally, the results are exported as Excel spreadsheets for each participant and region displaying exponent, offset, band's CF, PW, absolute power, relative power, and model fit error measures.

`/sart/sart_aperiodicfit_bp_analysis.ipynb` - importing clean SART EEG (.fif) files, calculating Welch's PSD for a single channel for GO and NO-GO tasks for post-event time period and post-event-minus-ERP (see the notebook for more details), estimate the aperiodic 1/f-like component of the spectra with specparam (FOOOF) algorithm, and export the aperiodic component's parameters (i.e., exponent and offset). Remove the aperiodic component of the spectra (i.e., flatten the spectra) and find absolute and relative theta (or any other band of interest) power. Finally, the results are exported as Excel spreadsheets for each participant displaying exponent, offset, band's absolute power, relative power, and model fit error measures.


### ERP analysis
`/auditory_oddball/oddball_erp_analysis.ipynb` - importing clean auditory oddball task EEG (.fif) files, averaging the epochs (i.e., evoked signal), finding peaks and identifying them based on pre-defined ERP time windows, prompting the user with evoked signal with identified ERPs providing manual ERP detection using min/max voltage detection in user-specified time windows. This semi-automated process will be done through all the participants and then ERPs' absolute amplitudes, latencies and peak-to-peak amplitudes are exported for each participant for a single channel of interest as an Excel spreadsheet.

`/sart/sart_erp_analysis.ipynb` - importing clean SART EEG (.fif) files, averaging the epochs (i.e., evoked signal), finding peaks and identifying them based on pre-defined ERP time windows, prompting the user with evoked signal with identified ERPs providing manual ERP detection using min/max voltage detection in user-specified time windows. This semi-automated process will be done through all the participants and then ERPs' absolute amplitudes, latencies and peak-to-peak amplitudes are exported for each participant for a single channel of interest as an Excel spreadsheet.


### Complexity & Entropy analysis
`/resting_state/resting_entropy_complexity.ipynb` - importing clean resting state EEG (.fif) files, resampling to 256 Hz, calculating Lempel-Ziv complexity (LZC) and Multiscale Sample Entropy (MSE) and export the results as an Excel spreadsheet with measures calculated for all participants.

### Data visualisation
`/resting_state/resting_data_visualisation.ipynb` (*outdated*) - reads all PSD Excel files and compares different conditions with each other using a statistical test of choice (currently only paired t-test and Wilcoxon signed-rank test), thereby giving information about statistically significant bandpower changes in various locations (regions and channels) between different conditions. Finally, various plotting functions can be used to plot the results. Also possible to import other measures to find correlation between the change of variables within timepoints; scatter plot function for Spearman or Pearson.

`/auditory_oddball/oddball_data_visualisation.ipynb` - importing clean auditory oddball task EEG (.fif) files, averaging the epochs (i.e., evoked signal), and average all the evoked signal across all participants (possible to exclude specific participants). The grand average will be plotted and can be done for comparing different conditions or timepoints.

`/sart/sart_data_visualisation.ipynb` - importing clean SART EEG (.fif) files, averaging the epochs (i.e., evoked signal), and average all the evoked signal across all participants (possible to exclude specific participants). The grand average will be plotted and can be done for comparing different conditions or timepoints.


### TO-DO
- Other time-domain waveform analysis etc (e.g., using bycycle package)
- Restructure the project and maybe turn into a package (DM if you could help!)


## Study notebooks
Find in folder `/studies`. Different studies/publications that have used this pipeline for the EEG analysis and their corresponding notebooks.

### Published work
`OKTOS_rsEEG_classic_bp.ipynb` - Anijärv et al. 2023. "Spectral Changes of EEG Following a 6-Week Low-Dose Oral Ketamine Treatment in Adults With Major Depressive Disorder and Chronic Suicidality". International Journal of Neuropsychopharmacology, Volume 26, Issue 4, April 2023, Pages 259–267, https://doi.org/10.1093/ijnp/pyad006

`Campbell_Resting_EEG_Sustained_Attention_Healthy_Ageing_Cross_Sectional_LEISURE.ipynb` - Campbell et al., 2024, Resting-state EEG correlates of sustained attention in healthy ageing: Cross-sectional findings from the LEISURE study. Neurobiology of Aging, Volume 144, September 2023, Pages 68–77, https://doi.org/10.1016/j.neurobiolaging.2024.09.005

## Requirements
The data processing and analysis is tested with Biosemi 32-channel EEG set. I recommend to create a [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the environment.yml file in this repository. However, down below you can see all the required libraries across parts of the pipeline in case you want to use only a specific notebook.

`conda env create -n EEG-pyline -f environment.yml`

If you want to install all the necessary packages separately then these four installs will cover all the packages.

`conda install -c conda-forge mne`
`conda install -c conda-forge autoreject`
`conda install -c nclibz statannotations`
`conda install -c conda-forge fooof`
`conda install -c anaconda openpyxl`

### Pre-processing:
- MNE
- AutoReject

### Spectral analysis + ERP analysis:
- MNE
- specparam (fooof)
- Pandas
- NumPy
- SciPy
- Matplotlib

### Complexity & Entropy analysis
- MNE
- Pandas
- NumPy
- neurokit2

### Data visualisation:
- MNE
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib
- Statannotations

## Citation
If you are using this project in your EEG study, it would be much appreciated if you could cite this repository in your work. See `CITATION.cff` or [![DOI](https://zenodo.org/badge/495300654.svg)](https://zenodo.org/badge/latestdoi/495300654) for information.

If you are using any specific study notebook, then additionally please add citation to the corresponding publication in your article.

## References
[1] Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hämäläinen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013. doi:10.3389/fnins.2013.00267.

[2] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, “Automated rejection and repair of bad trials in MEG/EEG.” In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[3] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.

[4] McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.

[5] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

[6] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

[7] Waskom M, et al. mwaskom/seaborn: v0.8.1 (September 2017) [Internet]. Zenodo; 2017. Available from: https://doi.org/10.5281/zenodo.883859

[8] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.

[9] Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD, Knight RT, Shestyuk A, Voytek B (2020). Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 23, 1655-1665. DOI: 10.1038/s41593-020-00744-x

[10] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H.,
Schölzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing.
Behavior Research Methods, 53(4), 1689–1696. https://doi.org/10.3758/s13428-020-01516-y
