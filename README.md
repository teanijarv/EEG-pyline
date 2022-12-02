# EEG pipeline for resting and task EEG pre-processing and analyses
This is an EEG pipeline used at the USC's Thompson Institute created by Toomas Erik Anijärv.

## Notebook templates


### Pre-processing
`resting_preprocessing.ipynb` - importing raw resting state EEG (.bdf) files, re-referencing, applying bandpass (e.g., 1-30 Hz) FIR filter, cropping the signal, removing EOG noise with SSP, dividing signal into equal-sized epochs (e.g., 5 seconds), performing artefact rejection with Autoreject algorithm, exporting the cleaned EEG signals (.fif).

`oddball_erp_preprocessing.ipynb` - importing raw auditory oddball task EEG (.bdf) files, re-referencing, applying bandpass (e.g., 1-15 Hz) FIR filter, removing EOG noise with SSP, finding event points including target tone followed by participant's button press, dividing the signal into 1-second epochs (i.e., -0.2-0s is pre-stimulus, 0s is stimulus or target tone, 0-0.8s is post-stimulus), performing artefact rejection with Autoreject algorithm, exporting the cleaned EEG signal (.fif).


### Spectral analysis
`resting_classic_bp_analysis.ipynb` - importing clean resting state EEG (.fif) files, calculating Welch's power spectrum density (PSD) for each region of interest (e.g., averaging channels together or channel-by-channel) and divides the PSD estimates to pre-defined frequency bands (e.g., delta, theta, alpha, beta). Optionally it is possible to calculate the (alpha) asymmetry using channels from each hemispheres and also find relative band powers by dividing the absolute band power value with broadband power. Finally, the results are exported as Excel spreadsheets for each band power, region and participant.

`resting_aperiodicfit_bp_analysis.ipynb` - importing clean resting state EEG (.fif) files, calculating Welch's PSD for each region of interest, estimate the aperiodic 1/f-like component of the spectra with specparam (FOOOF) algorithm, and export the aperiodic component's parameters (i.e., exponent and offset). Remove the aperiodic component of the spectra (i.e., flatten the spectra) and use pre-defined band of interest (e.g., alpha) to find its peak parameters (i.e., center frequency and peak width) and absolute and relative PSDs. Finally, the results are exported as Excel spreadsheets for each participant and region displaying exponent, offset, band's CF, PW, absolute power, relative power, and model fit error measures.


### ERP analysis
`oddball_erp_analysis.ipynb` - importing clean auditory oddball task EEG (.fif) files, averaging the epochs (i.e., evoked signal), finding peaks and identifying them based on pre-defined ERP time windows, prompting the user with evoked signal with identified ERPs providing manual ERP detection using min/max voltage detection in user-specified time windows. This semi-automated process will be done through all the participants and then ERPs' absolute amplitudes, latencies and peak-to-peak amplitudes are exported for each participant for a single channel of interest as an Excel spreadsheet.


### Data visualisation
`resting_data_visualisation.ipynb` (*not updated*) - reads all PSD Excel files and compares different conditions with each other using a statistical test of choice (currently only paired t-test and Wilcoxon signed-rank test), thereby giving information about statistically significant bandpower changes in various locations (regions and channels) between different conditions. Finally, various plotting functions can be used to plot the results. Also possible to import other measures to find correlation between the change of variables within timepoints; scatter plot function for Spearman or Pearson.

`oddball_data_visualisation.ipynb` - importing clean auditory oddball task EEG (.fif) files, averaging the epochs (i.e., evoked signal), and average all the evoked signal across all participants (possible to exclude specific participants). The grand average will be plotted and can be done for comparing different conditions or timepoints.


### TO-DO
- Non-linear analysis (e.g., Higuchi fractal dimension)
- Other time-domain waveform analysis etc
- A lot of code optimization still needs to be done...


## Study notebooks
Different studies/publications that have used this pipeline for the EEG analysis and their corresponding notebooks.

`OKTOS_rsEEG_classic_bp.ipynb` - Anijärv, Can et al. 2022. "Spectral changes of EEG following a 6-week low-dose oral ketamine treatment in adults with major depressive disorder and chronic suicidality". [Under review]
`LEISURE_rsEEG_aperiodic_activity.ipynb` - Campbell et al. 2022. "1/f and Attention: Examining the Relationship Between Attention and Aperiodic Neural Activity in Resting-State EEG in Ageing". [Manuscript in progress]

**Coming soon:**

`LABS_rsEEG_classic_bp.ipynb`

`OKTOS_aoEEG_erp_analysis.ipynb`

`LEISURE-LABS_rsEEG_aperiodic+iaf.ipynb`

## Requirements
The data processing and analysis is tested with Biosemi 32-channel EEG set. I recommend to create a [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the environment.yml file in this repository. However, down below you can see all the required libraries across parts of the pipeline in case you want to use only a specific notebook.

`conda env create -n EEG-pipeline-TI -f environment.yml`

If you want to install all the necessary packages separately then these four installs will cover all the packages.

`conda install -c conda-forge mne`
`conda install -c conda-forge autoreject`
`conda install -c nclibz statannotations`
`conda install -c conda-forge fooof`
`conda install -c anaconda openpyxl`

### Pre-processing:
- MNE
- AutoReject

### Spectral analysis:
- MNE
- specparam (fooof)
- Pandas
- NumPy
- SciPy
- Matplotlib

### Data visualisation:
- MNE
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib
- Statannotations

## Citation
If you are using this project/pipeline in your EEG analysis, it is not mandatory to cite or refer to this repository, but it would be much appreciated if you did. *Coming soon on how to cite to this repo.*

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
