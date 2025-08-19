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
    tif_file      = tif_override or "InputFiles/TestInputs/VB-APS-SSAO-6_25C_Map-AO_000176.avg.tiff"
    mask_thresh   = 4e2 # threshold value for the image mask
    num_azim_bins = 120 # number of azimuthal bins around the data
    q_min_nm1     = 14.0 # q_0 for binning of the data
    npt_rad       = 2048 # number of radial bins (~2-3x the radial pixel count)
    delta_tol     = 0.1 # default q-search width tolerance in nm^-1
    wavelength_nm = 0.1729786687 # [nm] X-ray wavelength
    solved_strain_components = 5 # This is the number of strain components to solve for in the system. # 3 = biaxial; 5 = biaxial w/ shear; 6 = all components
    initial_q_guesses = [
        17.957430, 
        24.499120, 
        26.267714, 
        29.972252, 
        35.923938, 
        39.032177, 
        41.355087, 
        44.509848] # initial guesses for peak fitting [nm^-1] for Alumina
    # initial_q_guesses = [ # initial guesses for peak fitting [nm^-1] for calibrant
    #     15.111021, 
    #     21.370204, 
    #     26.171220, 
    #     30.222884, 
    #     33.791341, 
    #     37.018340, 
    #     42.747420, 
    #     45.341304] 
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
        logger=file_logger)

    # Creates output directory for the χ data if there isn't one already
    chi_path = fl.create_directory(f"{output_path}/ChiOutput", logger=file_logger)

    # Bins and integrates the image data, then outputs the q vs χ (azimuth) data
    I2d, q, chi = fl.integrate_2d(
        ai, data, mask,
        num_azim_bins=num_azim_bins,
        q_min=q_min_nm1,
        npt_rad=npt_rad,
        output_dir=chi_path,
        logger=file_logger
    )

    # Fits the q vs χ data to the Pseudo-Voigt function to find the peak centroids for each bin and ring
    q_vs_chi, q_chi_path = fl.fit_peaks_with_initial_guesses(
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

    # Fit the full strain tensor using least squares and the full tensor model
    strain_tensor_components, strain_list, q0_list, strain_vs_chi_file = fl.fit_lattice_cone_distortion(
        q_data=q_vs_chi,
        q0_list=initial_q_guesses,
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