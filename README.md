# PyStrain2DXRD: Python workflow for calculating Strain using 2-Dimensional XRD
DOI: 10.5281/zenodo.17605591

These are the instructions for the workflow provided in this git repository. Please read all the way to the bottom before using the scripts the first time. 

# Installation Instructions
>[!NOTE]
>This repository automatically installs the pyFAI library using pip
>For more information about this powerful library, see the [pyFAI documentation](https://pyfai.readthedocs.io/en/stable/index.html)

Below are the instructions to install the pyFAI library and all other required libraries for this workflow:
>[!TIP]
>It is also **HIGHLY** recommended to run this workflow within its own Python virtual environment. Please see [this link](https://docs.python.org/3/tutorial/venv.html) to learn more about virtual environments, or look them up online. There are lots of useful resources on the topic.

## MacOS/Linux
1. Open a terminal session
2. Navigate to the parent directory into which you want to clone the repository
 ```
 cd /path/to/parent/directory
 ```
3. Clone the git repository
 ```
 git clone https://github.com/aaronzwells/Analysis_pyFAI.git
 ```
4. Go to the cloned repository
 ```
 cd Analysis_pyFAI
 ```
5. Create the virtual environment
 ```
 python3 -m venv .venv
 ```
6. Activate the virtual environment associated with the repository
 ```
 source .venv/bin/activate
 ```
7. Install the required packages
 ```
 pip install -r pyFAI/requirements.txt
 ```
## Windows
1. Open Command Prompt or PowerShell
2. Navigate to the parent directory
 ```
 cd C:\path\to\parent\directory
 ```
3. Clone the git repository
 ```
 git clone https://github.com/aaronzwells/Analysis_pyFAI.git
 ```
4. Go to the cloned repository
 ```
 cd Analysis_pyFAI
 ```
5. Create the virtual environment
 ```
 python -m venv .venv
 ```
6. Activate the virtual environment
 ```
 .venv\Scripts\activate
 ```
7. Install the required packages
 ```
 pip install -r pyFAI\requirements.txt
 ```

# OVERVIEW OF DIRECTORIES

## PARENT DIRECTORY

**Analysis_pyFAI/** is the main directory where all inputs and output subdirectories are located, as well as the script files (*.py). 

## USER DIRECTORIES

**InputFiles/** is the directory where all user-provided input images are housed. It is **_not included_ in the git repository**, and should be created by the user. It was excluded due to the size of the folders.

__OutputData/OutputFiles*/__ directories are generated automatically during script generation. These are named with the following format: "OutputFiles\_\<kind_of_output\>\_\<sample_name\>\_\<date_and_time\>"

**OutputMaps_*/** directories are created to house the final stress and strain maps. Each map directory is time stamped and uses the above ^^ naming convention.

**PeakFinding/** is where the reference peak data is stored after running 1.FindingRefPeaks.py

**ValidationOutputFiles/** is where the data from 2.StrainCalc-SingleImage.py are saved

**calibration/** is the directory where pyFAI calibration data should be stored. (The user is responsible for storing the data here. It does not auto-save to this directory.) This data includes *<ins>all</ins> .npt, .poni, .json files* related to calibration. It is also recommended that *.tif* calibration images be saved to this folder, as it is a convenient way to keep everything nicely contained.

## AUXILIARY DIRECTORIES

**pyFAI/** is the directory where the pyFAI program is installed. For details on this installation see the [PyFAI Documentation](https://pyfai.readthedocs.io/en/stable/index.html) page.

**__pycache__/** is a folder that houses all cached data from the batch processing steps. There is no need for the user to ever interact with this directory as it is auto-generated.

# WORKFLOW OVERVIEW

Before executing the workflow, 1.FindingRefPeaks.py can be used to run the peak fitting math.  This is useful for ensuring the peaks are being correctly fit to given input data. *FunctionLibrary.py* contains all the functions used and called in the below scripts. 

>[!NOTE]
>The main workflow scripts are numbered and may be referred to as "*Script 1*", "*Script 2*", etc.

## WORKFLOW OUTLINE & SUMMARY

1. The first true workflow script is `1.GettingPeakLocs.py`. This script allows the user to input their calibration *.poni* file and a representative *.tif* or *.tiff* file and have PyFAI determine good initial guesses for the first 9 peaks by fitting to the Pseudo-Voigt function. It then outputs this data to a folder, named after your input file, within the `ValidationOutputFiles/PeakFinding/` directory. 

2. The second workflow script is `2.StrainCalcs-SingleImage.py`. **This is the script that finds the reference q0 values for all future strain calculations.** If using it to find the reference, set `q0_reference_file` to `None`. Otherwise, provide the path to the already found `q0_reference_file`. It is recommended to input the values from *Script 1* for `initial_q_guesses`. Other parameters should be set via the user's experimental conditions and desired processing parameters.

>[!TIP]
>Using relative paths when updating script variables is <ins>accepted and encouraged</ins> to keep the main directory organized. 

3. The third workflow script is `3.StrainCalcs-BatchProcess.py`. This script is the main data analysis script. It requires similar inputs (housed in `config`, lines 208-243) as *Script 2*, though it only needs the parent directory in which the input TIFs are housed. This script will output a significant amount of data housed in an `OutputFiles_Data_*/` directory matching the **user-defined** sample name prefix in the variable called `SampleName`. 

4. The mapping scripts are either `4.MappingStrain.py` and `5.MappingStress.py`. These two scripts map the strain based upon the generated `strain_tensor_summary.json` from *Script 3*. The outputs are saved to an output directory using the same naming conventions as the outputs from *Script 3*. 

>[!NOTE]
>*Script 5* requires the input of material properties and is only set up to solve for perfectly elastic materials using Hooke's Law.

## Miscellaneous Scripts
All of these scripts contain the prefix `Utility-` and can be used or adapted to perform different functions. The first three should work, but `Utility-ReconstructingRingsFromStress.py` was never completed.

## REQUIRED USER-DEFINED VARIABLES

>[!NOTE]
>See the comments in the code for a definition of each variable

### Script 1 - FindingRefPeaks
Lines 8-9
* `poni_file` 
* `tif_file`

### Script 2 - StrainCalcs-SingleImage
Lines 35-63
* `poni_file`
* `q0_reference_file`
* `tif_file`
* `mask_file`
* `save_chi_files`
* `save_adjusted_tif`
* `mask_thresh`
* `num_azim_bins`
* `q_min_nm1`
* `npt_rad`
* `delta_tol`
* `wavelength_nm`
* `solved_strain_components`
* `MAD_threshold`
* `initial_q_guesses`
* `tol_array`
* `eta0`

### Script 3 - StrainCalcs-BatchProcess
Lines 208-243
* `input_dir`
* `sampleName`
* `poni_file`
* `q0_reference_file`
* `mask_file`
* `save_chi_files`
* `plot_q_vs_chi`
* `plot_strain_vs_chi`
* `save_adjusted_tif`
* `num_jobs_parallel`
* `mask_thresh`
* `num_azim_bins`
* `q_min_nm1`
* `npt_rad`
* `delta_tol`
* `wavelength_nm`
* `solved_strain_components`
* `MAD_threshold`
* `initial_q_guesses`
* `tol_array`
* `eta0`
* `min_rsquared`

### Script 4 - MappingStrain
Lines 8-27
* `json_path`
* `sample_name`
* `solved_strain_components`
* `n_steps_x`
* `n_steps_y`
* `dX`
* `dY`
* `pixel_size_map`
* `start_xy`
* `gap_mm`
* `map_offset_xy`
* `trim_negative_xy`
* `map_x_limits`
* `map_y_limits`
* `color_limit_window`
* `colorbar_scale`
* `colorbar_bins`
* `title_and_labels`

### Script 5 - MappingStress
Lines 15-38
* `json_path`
* `sample_name`
* `youngs_modulus`
* `poissons_ratio`
* `n_steps_x`
* `n_steps_y`
* `dX`
* `dY`
* `pixel_size_map`
* `start_xy`
* `gap_mm`
* `map_offset_xy`
* `trim_negative_xy`
* `map_x_limits`
* `map_y_limits`
* `color_limit_window`
* `colorbar_scale`
* `map_name_pfx`
