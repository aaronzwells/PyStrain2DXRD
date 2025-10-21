These are the instructions for the workflow provided in this git repository. Please read all the way to the bottom before using the scripts the first time. 

# Installation Instructions
>[!NOTE]
>This repository automatically installs the pyFAI library using pip
>For more information about this powerful library, see the [pyFAI documentation](https://pyfai.readthedocs.io/en/stable/index.html)

Below are the instructions to install the pyFAI library and all other required libraries for this workflow:
>[!TIP]
>It is also **HIGHLY** recommended to run this workflow within its own Python virtual  environment. Please see [this link](https://docs.python.org/3/tutorial/venv.html) to learn more  about virtual environments, or look them up online. There are lots of useful resources on the  topic.

## MacOS/Linux
1. Open a terminal session
1. Navigate to the parent directory into which you want to clone the repository
```
cd /path/to/parent/directory
```
1. Clone the git repository
```
git clone https://github.com/aaronzwells/Analysis_pyFAI.git
```
1. Go to the cloned repository
```
cd Analysis_pyFAI
```
1. Create the virtual environment
```
python3 -m venv .venv
```
1. Activate the virtual environment associated with the repository
```
source .venv/bin/activate
```
1. Install the required packages
```
pip install -r pyFAI/requirements.txt
```
## Windows
1. Open Command Prompt or PowerShell
1. Navigate to the parent directory
```
cd C:\path\to\parent\directory
```
1. Clone the git repository
```
git clone https://github.com/aaronzwells/Analysis_pyFAI.git
```
1. Go to the cloned repository
```
cd Analysis_pyFAI
```
1. Create the virtual environment
```
python -m venv .venv
```
1. Activate the virtual environment
```
.venv\Scripts\activate
```
1. Install the required packages
```
pip install -r pyFAI\requirements.txt
```

# OVERVIEW OF DIRECTORIES

## PARENT DIRECTORY

**Analysis_pyFAI/** is the main directory where all inputs and output subdirectories are  located, as well as the script files (*.py). 

## USER DIRECTORIES

**InputFiles/** is the directory where all user-provided input images are housed. It is **_not included_ in  the git repository**, and should be created by the user. It was excluded due to the size of the  folders.

**OutputData/** is where the batch process data is output

**PeakFinding/** is where the reference peak data is stored after running 1.FindingRefPeaks.py

**ValidationOutputFiles/** is where the data from 2.StrainCalc-SingleImage.py are saved

**calibration/** is the directory where pyFAI calibration data should be stored. (The user is  responsible for storing the data here. It does not auto-save to this directory.) This data  includes *<ins>all</ins> .npt, .poni, .json files* related to calibration. It is also  recommended that *.tif* calibration images be saved to this folder, as it is a convenient way  to keep everything nicely contained.

## LIBRARY DIRECTORIES

**pyFAI/** is the directory where the pyFAI program is installed. For details on this  installation see the [PyFAI Documentation](https://pyfai.readthedocs.io/en/stable/index.html)  page.

## SCRIPT-GENERATED DIRECTORIES

**OutputData/OutputFiles*/** directories are generated automatically during script generation. These are named  with the following format:  "OutputFiles\_\<kind_of_output\>\_\<sample_name\>\_\<date_and_time\>"

**OutputMaps_*/** directories are created to house the final stress and strain maps. Each map directory is time  stamped and uses the above ^^ naming convention.

**__pycache__/** is a folder that houses all cached data from the batch processing steps. There  is no need for the user to ever interact with this directory as it is auto-generated.

# WORKFLOW OVERVIEW

Before executing the workflow, ValidationScript.py can be used to run the peak fitting math.   This is useful for ensuring the peaks are being correctly fit to given input data.  *FunctionLibrary.py* contains all the functions used and called in the below scripts. 

>[!NOTE]
>The main workflow scripts are numbered and may be referred to as "*Script 1*", "*Script 2*", etc.

## WORKFLOW OUTLINE & SUMMARY

>[!CAUTION]
>This section needs to be rewritten to reflect recent updates.

1. The first true workflow script is `1.GettingPeakLocs.py`. This script allows the user to  input their calibration *.poni* file and a representative *.tif* or *.tiff* file and have PyFAI  determine good initial guesses for the first 9 peaks. It then outputs this data to a folder,  named after your input file, within the `ValidationOutputFiles/PeakFinding/` directory. 

2. The second workflow script is `2.MainPipeline-noBatch.py`. This script allows the user to  test their initial guesses from *Script 1* by adjusting the values in the  `initial_q_guesses` and (if necessary) `tol_array` variables. The user should also update the  calibration and *.tif* file paths. 

>[!TIP]
>Using relative paths when updating script variables is <ins>accepted and encouraged</ins> to  keep the main directory organized. 

3. The third workflow script is 3.MainPipeline-BatchProcess.py. This script is the main data  analysis script. It requires the same inputs as *Script 2*, though the input  should simply be the path to directory where the images for analysis will be held. This script  will output a significant amount of data housed in an `OutputFiles_Data_*/` directory matching  the **user-defined** sample name prefix in the variable called `SampleName`. 

4. The final workflow script is either `4a.MappingStrain-Continuous.py` or  `4b.MappingStrain-nonContinuous.py`. These two scripts map the strain based upon  the generated data from *Script 3*. The outputs are saved to an output directory using the same  naming conventions as the outputs from *Script 3*. 

>[!CAUTION]
>*Script 4b* has not been completed yet. Do not use until finalized.

## REQUIRED USER-DEFINED VARIABLES

>[!CAUTION]
>This section needs to be rewritten to reflect recent updates.

>[!NOTE]
>See the comments in the code for a definition of each variable

### Script 1
Lines 8-9
* `poni_file` 
* `tif_file`

### Script 2
Lines 35-53
* `poni_file`
* `tif_file`
* `mask_thresh`
* `num_azim_bins`
* `q_min_nm1`
* `npt_rad`
* `initial_q_guesses`
* `tol_array`

### Script 3
Lines 13, 14, 35, 37-45
* `input_dir` <-- input directory
* `sample_name`
* `poni_file`
* `tif_file`
* `mask_thresh`
* `num_azim_bins`
* `q_min_nm1`
* `npt_rad`
* `initial_q_guesses`
* `tol_array`

### Script 4a
Lines 5-9
* `json_path`
* `sample_name`
* `n_rows`
* `n_cols`
* `pixel_size`

### Script 4b
>[!CAUTION]
>This script has not been completed yet. Do not use until finalized.
