"""Azimuthal binning of data and fitting cone distortion for a single pattern """

import logging
import FunctionLibrary as fl
import time
import os
import numpy as np
from joblib import Parallel, delayed
import glob
from tqdm import tqdm

#Sample 6: February Dataset
#Samples 5, 3, 2: October Dataset

# --- Logger Setup ----------------------------------------------------------
def setup_logger(log_path, logger_name=None):
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add StreamHandler for console output
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# --- Main pipeline -------------------------------------------------------
def nobatch_main_pipeline(tif_override=None, batch_output_dir=None, output_tensor_path=None):

    #Sample 6: February Dataset
    #Samples 5, 3, 2: October Dataset

    start_time = time.time()
    
    #OCTOBER 2025 BEAMTIME CALIBRATION PARAMETERS: 
    # poni_file = "calibration/Calibration_Oct25_ceria_900mm_linkam_30C_att000_0006091.poni" # calibration PONI file. I used CeO2 
    # detector_type = "Pilatus" # "Pilatus" or "GE"
    # mask_file = "calibration/pilatus_mask.msk" # Either "path/to/your/mask.tif" or None
    
    #FEBRUARY 2025 BEAMTIME CALIBRATION PARAMETERS: 
    #This calibration gives ~0.03%-0.04% error from known ceria d-spacings. calibration/Calibration_LaB6_100x100_3s_r8_mod2_BAD.poni was 0.67-0.69% 
    poni_file = "calibration/Calibration_Feb25_ceria_1145mm_25C_att000_000112.poni" # calibration PONI file. Ben used CeO2
    detector_type = "GE" # "Pilatus" or "GE"
    mask_file = None 
    
    #The single image you are analyzing for strain data 
    #OCTOBER CALIBRANT
    # tif_file      = "InputFiles/Oct2025_linkam_temperature_calib/ceria_900mm_linkam_30C_att000/ceria_900mm_linkam_30C_att000_0006092.tif"

    #FEBRUARY CERIA ONLY CALIBRANT
    tif_file      = "InputFiles/Feb2025_Calibrant_Patterns/Feb2025_ceria_71p676keV_1145mm_100x100_3s_000112.avg.tiff"  


    #FEM Zero strain position of the current map. Use None if you are analyzing the zero strain position
    q0_reference_file = None #"ValidationOutputFiles/VB-APS-SSAO-6_25C_Map-AO_000304_ref/q0_vs_chi_FITTED.txt" 

    
    #ORIGINAL SCRIPT PARAMETERS
    save_chi_files = True # this determines whether every q vs chi bin dataset is saved as a separate file or if the file writing is skipped
    save_txt_for_fityk = True # Option to save a clean .txt file in a separate folder for Fityk scripting, to compare to Aaron's fitting code
    save_adjusted_tif = True
    mask_thresh   = None # Minimum threshold value for the image mask
    autocontrast_sensitivity = 0.5 # Defines the upper and lower bounds of the autocontrast; smaller is a more narrow intensity band
    num_azim_bins = 120 # number of azimuthal bins around the data (so each bin is 360/num_azim_bins degrees wide)
    q_min_nm1     = 14.0 # q_0 for binning of the data
    npt_rad       = 1100 # number of radial bins. KEEP BELOW THE MAX. RADIAL PIXEL COUNT TO AVOID MASKING BUGS. 
    delta_tol     = 0.1 # default q-search width tolerance in nm^-1
    wavelength_nm = 0.1729786687 # [nm] X-ray wavelength
    solved_strain_components = 5 # This is the number of strain components to solve for in the system. # 3 = biaxial; 5 = biaxial w/ shear; 6 = all components
    MAD_threshold = 2 # Threshold for median absolute deviation (MAD) filtering

    #Examine bins: Now a bare bones option to just look at the plotted binned data to make sure "2d" integration looks reasonable
    # If you want to visualize the binned data, set this to True. Should not typically be needed now that the intensity issue is corrected. 
    examine_bins = True 

    # initial_q_guesses = [ # February 2025 Al2O3 with Aaron calibration (positions may not be accurate)
    #             17.961188,
    #             24.500613,
    #             26.267830,
    #             29.974002,
    #             35.926353,
    #             39.034769,
    #             44.513621,
    #             45.514461
    #         ]

    # initial_q_guesses = [ # October 2025 Al2O3 (These peak positions should ultimately be correct, based on calibrant matching)
    #             18.103087,
    #             24.677268,
    #             26.458500,
    #             30.203330,
    #             36.188437,
    #             39.321810,
    #             44.830282,
    #             45.838482
    #         ]

    # initial_q_guesses = [ # February 2025 CeO2 Calibrant, Room Temp (These are WRONG ceria positions, but using them to run the script
    #                         #to verify binned intensity issue is pervasive)
    #                 19.973575,	
    #                 23.063715,	
    #                 32.620258,	
    #                 38.253129,	
    #                 39.954924,	
    #                 50.284085,	
    #                 51.591647,	
    #                 56.521486,	
    #         ]
    
    initial_q_guesses = [ # October 2025 CeO2 Calibrant, Room Temp. Using the same initial guesses for Feb. 2025 (Frame 112) after corrected cal. 
                20.108632,
                23.220192,
                32.840341,
                38.507202,
                40.219511,
                50.611331,
                51.926886,
                56.885665
            ]
  
    tol_array   = np.array([ # tolerance values for q when searching for a peak to fit [nm^-1] for calibrant
        [1, 1, 1, 1, 1, 1, 1, 1], # larger q
        [1, 1, 1, 1, 1, 1, 1, 1]]) # smaller q
    eta0          = 0.5
    
    # This removes the file extension and .avg from the end of the averaged image files
    filename = fl.remove_filename_extension(tif_file)
    
    # Creates an output directory of the same name as the mapping image to store all the data for that map image location
    outputPath = os.path.join("ValidationOutputFiles", filename)
    output_path = fl.create_directory(outputPath, logger=None)
    print(f"[INFO] Output Path is {output_path}")

    # --- Setup file logger for pipeline ---
    log_path = os.path.join(output_path, f"{filename}_pipeline.log")
    file_logger = setup_logger(log_path, logger_name=filename)

    # # If using the validation output as the initial q guesses, this converts that 2theta peak location data to q-space
    # q_peak_locs = fl.convert_2theta_to_q(
    #     "AdditionalFiles/FxnValidation/FitPeakLocations-Al2O3.txt", 
    #     wavelength_nm=wavelength)               
    # print(q_peak_locs)

    # Initializes the pyFAI integrator and imports the calibration parameters from the poni file.
    # BS June 2026: Corrected critical issue, now returning unaltered image data.
    ai, data, mask = fl.load_integrator_and_data(
        poni_file,
        tif_file,
        output_path=output_path,
        detector_type=detector_type,
        mask_file=mask_file,
        mask_threshold=mask_thresh,
        logger=file_logger,
        save_adjusted_tif=save_adjusted_tif,
        autocontrast_sensitivity=autocontrast_sensitivity)

    # Creates output directory for the binned data if there isn't one already
    binned_path = fl.create_directory(f"{output_path}/BinnedOutput", logger=file_logger)

    # Bins and integrates the image data, then outputs the q vs χ (azimuth) data
    I2d, q, chi = fl.integrate_2d(
        ai, data, detector_type, mask, 
        num_azim_bins=num_azim_bins,
        q_min=q_min_nm1,
        npt_rad=npt_rad,
        output_dir=binned_path,
        save_chi_files=save_chi_files,
        save_txt_for_fityk=save_txt_for_fityk,
        logger=file_logger
    )

    #Performs an analysis similar to script 1 on each bin, if examine_bins is true
    if examine_bins:
        fl.plot_binned_patterns_from_2d_integration(I2d, q, chi, output_dir=binned_path, logger=file_logger)
        #The function puts the plots in a subdirectory in the same BinnedOutput folder

    # Fits the q vs χ data to the Pseudo-Voigt function to find the peak centroids for each bin and ring
    q_vs_chi, q_vs_chi_errors, q_chi_path = fl.fit_peaks_with_initial_guesses(
        I2d, 
        chi,
        q, 
        initial_q_guesses, 
        delta_tol=delta_tol, 
        eta0=eta0, 
        delta_array=tol_array, 
        output_dir=output_path,
        logger=file_logger)
    
    fl.plot_q_vs_chi_stacked(
        file_path=q_chi_path,
        output_dir=output_path,
        dpi=600,
        plot=True,
        logger=file_logger)

    # Initialize the q0 data
    if q0_reference_file:
        try: 
            q0_chi_ref = np.loadtxt(q0_reference_file)
            file_logger.info(f"Loaded q0(chi) reference data from {q0_reference_file}")
        except Exception as e:
            file_logger.error(f"FATAL: Could not load q0 reference file: {e}")
            return
    else:
        # If no file is provided, create a "self-referenced" baseline
        # This uses the average q of each ring as its own reference
        file_logger.info("No q0_reference_file provided. Creating a self-referenced baseline.")
        # Calculate the mean q for each ring, ignoring any NaN values
        mean_q_per_ring = np.nanmean(q_vs_chi, axis=1, keepdims=True)
        # Create the reference array by repeating the mean value across all azimuthal bins
        q0_chi_ref = np.tile(mean_q_per_ring, (1, q_vs_chi.shape[1]))

    # Fit the full strain tensor using least squares and the full tensor model
    strain_tensor_components, strain_list, q0_list, strain_vs_chi_file = fl.fit_lattice_cone_distortion(
        q_data=q_vs_chi,
        q_errors=q_vs_chi_errors,
        q0_chi_data=q0_chi_ref,
        initial_q_guesses=initial_q_guesses,
        wavelength_nm=wavelength_nm,
        chi_deg=chi,
        psi_deg=None,
        phi_deg=None,
        omega_deg=None,
        num_strain_components=solved_strain_components,
        MAD_threshold=MAD_threshold,
        output_dir=output_path,
        dpi=600,
        plot=True,
        logger=file_logger)

    # Plots the strain vs chi plots
    fl.plot_strain_vs_chi_stacked(
        file_path=strain_vs_chi_file, 
        output_dir=output_path, 
        dpi=600, 
        plot=True,
        calibrant=True, # boolean that should be changed to True if running a calibrant specimen
        logger=file_logger)

    end_time = time.time()
    run_time = end_time - start_time
    print(f"[INFO] Run Time: {run_time} seconds")

if __name__ == "__main__":
    nobatch_main_pipeline()