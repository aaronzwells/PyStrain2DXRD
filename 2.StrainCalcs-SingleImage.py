import logging
import FunctionLibrary as fl
import time
import os
import numpy as np
from joblib import Parallel, delayed
import glob
from tqdm import tqdm


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
    start_time = time.time()
    poni_file     = "calibration/Calibration_LaB6_100x100_3s_r8_mod2.poni"
    q0_reference_file = None #"ValidationOutputFiles/VB-APS-SSAO-6_25C_Map-AO_000304/q_vs_chi_peaks.txt"
    tif_file      = tif_override or "InputFiles/Reference_0Strain_inputs/VB-APS-SSAO-6_30C_cool_Map-AO_001474.avg.tiff"
    save_chi_files = False # this determines whether every q vs chi bin dataset is saved as a separate file or if the file writing is skipped
    save_adjusted_tif = False
    mask_thresh   = 4e2 # threshold value for the image mask
    num_azim_bins = 120 # number of azimuthal bins around the data
    q_min_nm1     = 14.0 # q_0 for binning of the data
    npt_rad       = 2048 # number of radial bins (~2-3x the radial pixel count)
    delta_tol     = 0.1 # default q-search width tolerance in nm^-1
    wavelength_nm = 0.1729786687 # [nm] X-ray wavelength
    solved_strain_components = 5 # This is the number of strain components to solve for in the system. # 3 = biaxial; 5 = biaxial w/ shear; 6 = all components
    initial_q_guesses = [
                17.959886,
                24.497447,
                26.268024,
                29.970841,
                35.922285,
                39.032558,
                44.507226,
                45.509875
            ]
     
    tol_array   = np.array([ # tolerance values for q when searching for a peak to fit [nm^-1] for calibrant
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # larger q
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]) # smaller q
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
    ai, data, mask = fl.load_integrator_and_data(
        poni_file,
        tif_file,
        output_path=output_path,
        mask_threshold=mask_thresh,
        logger=file_logger,
        save_adjusted_tif=save_adjusted_tif)

    # Creates output directory for the χ data if there isn't one already
    chi_path = fl.create_directory(f"{output_path}/ChiOutput", logger=file_logger)

    # Bins and integrates the image data, then outputs the q vs χ (azimuth) data
    I2d, q, chi = fl.integrate_2d(
        ai, data, mask,
        num_azim_bins=num_azim_bins,
        q_min=q_min_nm1,
        npt_rad=npt_rad,
        output_dir=chi_path,
        save_chi_files=save_chi_files,
        logger=file_logger
    )

    # Fits the q vs χ data to the Pseudo-Voigt function to find the peak centroids for each bin and ring
    q_vs_chi, q_vs_chi_errors, q_chi_path = fl.fit_peaks_with_initial_guesses(
        I2d, 
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