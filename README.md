# EEG pipeline from pre-processing to results visualisation
This is an EEG pipeline used at the USC's Thompson Institute created by Toomas Erik Anijärv. It consists of pre-processing, spectral analysis, and data visualisation notebooks.

## Notebooks
**Pre-processing** is made to work with importing raw BDF files acquired with Biosemi 32-channel EEG cap, then filtering with 0.5-50 Hz bandpass filter, removing EOG noise with EOG-based channels, performing artefact rejection and augmentation, and finally exporting the clean EEG signals as FIF files.

**Spectral analysis** reads clean EEG (FIF files) signals for each subject, calculates power spectrum density (PSD) for each band of interest, does signal quality check (visual inspection), and exports the bandpowers regionally and channel-by-channel into Excel files.

**Data visualisation** reads all PSD Excel files and compares different conditions with each other using a statistical test of choice (currently only paired t-test and Wilcoxon signed-rank test), thereby giving information about statistically significant bandpower changes in various locations (regions and channels) between different conditions. Finally, plotting functions (`plot_topomaps_band`, `plot_boxplot_band`, `plot_boxplot_location`) can be used to plot the results.

### TO-DO
- Non-linear analysis (e.g., Higuchi fractal dimension)
- More time-domain analyses (e.g., ERP)

## Requirements
The data processing and analysis is tested with Biosemi 32-channel EEG set. I recommend to create a [conda environment](https://www.anaconda.com/distribution/) with all the dependencies using the environment.yml file in this repository. However, down below you can see all the required libraries across parts of the pipeline in case you want to use only a specific notebook.

`conda env create -n eeg-pipeline -f environment.yml`

### Pre-processing:
- MNE
- AutoReject

### Spectral analysis:
- MNE
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib

### Data visualisation:
- MNE
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib
- Statannotations

## References
[1] Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hämäläinen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013. doi:10.3389/fnins.2013.00267.

[2] Mainak Jas, Denis Engemann, Federico Raimondo, Yousra Bekhti, and Alexandre Gramfort, “Automated rejection and repair of bad trials in MEG/EEG.” In 6th International Workshop on Pattern Recognition in Neuroimaging (PRNI), 2016.

[3] Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.

[4] McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.

[5] Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

[6] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.

[7] Waskom M, et al. mwaskom/seaborn: v0.8.1 (September 2017) [Internet]. Zenodo; 2017. Available from: https://doi.org/10.5281/zenodo.883859

[8] J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
