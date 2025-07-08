These are the instructions for the workflow provided in this git repository. Please read all  the way to the bottom before using the scripts the first time. 

>[!IMPORTANT]
>This workflow does require an installation of PyFAI with the additional libraries added to the  original requirements.txt file for PyFAI. This revised requirements.txt file can be found at  `*/Analysis_pyFAI/pyFAI/requirements.txt`. 

>[!TIP]
>It is also **HIGHLY** recommended to run this workflow within its own Python virtual  environment. Please see [this link](https://docs.python.org/3/tutorial/venv.html) to learn more  about virtual environments, or look them up online. There are lots of useful resources on the  topic.

# OVERVIEW OF DIRECTORIES

## PARENT DIRECTORY

**Analysis_pyFAI/** is the main directory where all inputs and output subdirectories are  located, as well as the script files (*.py). 

## MAIN DIRECTORIES

**pyFAI/** is the directory where the pyFAI program is installed. For details on this  installation see the [PyFAI Documentation](https://pyfai.readthedocs.io/en/stable/index.html)  page.

**AdditionalFiles/** is the directory that houses extraneous data; it is primarily used for  `ValidationScript.py`

**calibration/** is the directory where pyFAI calibration data should be stored. (The user is  responsible for storing the data here. It does not auto-save to this directory.) This data  includes *<ins>all</ins> .npt, .poni, .json files* related to calibration. It is also  recommended that *.tif* calibration images be saved to this folder, as it is a convenient way  to keep everything nicely contained.

**InputFiles/** is the directory where all input images are housed. It is **_not included_ in  the git repository**, and should be created by the user. It was excluded due to the size of the  folders.

## SECONDARY DIRECTORIES

**OldScripts/** is the directory that houses early revisions or editions of the scripts in this  workflow. Most are obsolete, but may be useful in certain scenarios.

**ValidationOutputFiles/** is the directory that houses the output data from the validation  process, including all runs from `1.GettingPeakLocs.py` & `2.MainPipeline-noBatch.py`

## SCRIPT-GENERATED DIRECTORIES

**__pycache__/** is a folder that houses all cached data from the batch processing steps. There  is no need for the user to ever interact with this directory as it is auto-generated.

__OutputFiles*/__ directories are generated automatically during script generation. These named  with the following format:  "OutputFiles\_\<kind_of_output\>\_\<input_file_name\>\_\<date_and_time\>"

# WORKFLOW OVERVIEW

Before executing the workflow, ValidationScript.py can be used to run the peak fitting math.   This is useful for ensuring the peaks are being correctly fit to given input data.  *FunctionLibrary.py* contains all the functions used and called in the below scripts. 

>[!NOTE]
>The main workflow scripts are numbered and may be referred to as "*Script 1*", "*Script 2*", etc.

## WORKFLOW OUTLINE/SUMMARY

1. The first true workflow script is `1.GettingPeakLocs.py`. This script allows the user to  input their calibration *.poni* file and a representative *.tif* or *.tiff* file and have PyFAI  determine good initial guesses for the first 9 peaks. It then outputs this data to a folder,  named after your input file, within the `ValidationOutputFiles/PeakFinding/` directory. 

2. The second workflow script is `2.MainPipeline-noBatch.py`. This script allows the user to  test their initial guesses from *Script 1* by adjusting the values in the  `initial_q_guesses` and (if necessary) `tol_array` variables. The user should also update the  calibration and *.tif* file paths. 

>[!TIP]
>Using relative paths when updating script variables is <ins>accepted and encouraged</ins> to  keep the main directory organized. 

3. The third workflow script is 3.MainPipeline-BatchProcess.py. This script is the main data  analysis script. It requires the same inputs as *Script 2*, though the input  should simply be the path to directory where the images for analysis will be held. This script  will output a significant amount of data housed in an `OutputFiles_Data_*/` directory matching  the **user-defined** sample name prefix in the variable called `SampleName`. 

4. The final workflow script is either `4a.MappingStrain-Continuous.py` or  `4b.MappingStrain-nonContinuous.py`. These two scripts map the strain based upon  the generated data from *Script 3*. The outputs are saved to an output directory using the same  naming conventions as the outputs from *Script 3*. 

>[!CAUTION]
>*Script 4b* has not been completed yet. Do not use until finalized.

## REQUIRED USER-DEFINED VARIABLES

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
