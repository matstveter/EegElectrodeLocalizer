# EEG electrode detection from 3D scans

## Description
This tool is designed to automatically localize fiducials and EEG electrodes from a 3D scan. It was primarily 
developed to explore the feasibility of such a solution and to scientifically evaluate its performance. The article
can be found as a preprint here (https://www.biorxiv.org/content/10.1101/2024.06.27.600334v1) and all information about the pipeline and assumptions can be found there.

### IMPORTANT: This tool is initially designed for a single dataset using EEG caps from one supplier in two sizes. For compatibility with other datasets or suppliers' EEG caps, some code modifications might be necessary.

## Features
- Localize the fiducials 
- Localize the EEG electrodes
- Manual verification tool for both the fiducials and electrodes

## Minimum Requirements
- These are the absolute minimum requirements; the full procedure of the research can be found in the article.
- The minimum requirements for the Python packages are listed in the requirements.txt file.
- This code was created for detecting electrodes from ANT Neuro 126-channel EEG caps in large (blue) or medium (blue and red) sizes. It has not been tested in other locations or with other suppliers.
- The pipeline assumes one folder per subject. Each subject needs a "Model.obj" file for the mesh and a corresponding "Model.jpg" file for the texture, all in a {ID}.zip folder.
- For the automatic pipeline for detecting fiducials to work (primarily for large-scale data processing), the fiducials must be marked with white circular dots as specified in the article.

## How to run the code:
1. Install all required packages in requirements file
2. Specify the data in the config file: pos_3d/config/3d_config.ini
   - Absolute path to the dataset
   - If another template file than the ant 126 channel system. Put this file (.elc) in the pos_3d/config/ folder, and call it pos.elc.
   - The output will be in the project results folder, if another is wanted, this must be changed in the settings.py file
   - Other important variables:
     - set_fiducials_manually: This skips the detection of orientation and landmark detection. This is a good option to have to True to increase the possibility for the algorithm working on other datasets and suppliers.
     - verify_fiducials: Plots and shows the localized landmarks and allows for manual adjustments
     - visual_validation: Plots and shows the landmarks, and plot and shows the electrodes. This is wanted unless the algorithm is running on a large dataset
     - manual_adjustment_if_red_flag: Some operation raises a red-flag, this allows for manual adjustments regardless of other options being set to False. Suggests to have this set to True
     - use_detected_fiducials: Set to True, if the landmarks have been set during another run, and you want to skip the entire landmark process. Then the "path_detected_fiducials" has to be specified with the path to the folder.
3. python3 main.py


## Citation
@article {Tveter2024.06.27.600334,
	author = {Tveter, Mats and Tveitstol, Thomas and Nygaard, Tonnes and Perez T., Ana S. and Kulashekhar, Shrikanth and Bruna, Ricard and Hammer, Hugo L. and Hatlestad-Hall, Christoffer and Hebold Haraldsen, Ira R. J.},
	title = {EEG Electrodes and Where to Find Them: Automated Localization From 3D Scans},
	elocation-id = {2024.06.27.600334},
	year = {2024},
	doi = {10.1101/2024.06.27.600334},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/06/30/2024.06.27.600334},
	eprint = {https://www.biorxiv.org/content/early/2024/06/30/2024.06.27.600334.full.pdf},
	journal = {bioRxiv}
}
