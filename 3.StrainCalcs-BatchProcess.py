import os
import matplotlib
matplotlib.use('Agg') # Uses the Agg backend for matplotlib to prevent UI generation, which speeds up the image saving and prevents issues with threaded parallelizing
import logging
import time
import numpy as np
from joblib import Parallel, delayed
import glob
import json
import multiprocessing
import pyFAI, fabio
import FunctionLibrary as fl # This is the REQUIRED custom library for this analysis.

# Batch processing for all .tif files in the defined input directory
def batch_main_pipeline(config):
    n_jobs = config['num_jobs_parallel']
    sampleName = config['sampleName']
    input_dir = config['input_dir']
    
    # Setting up the time metrics and creating the parent output directory
    start_time = time.time()
    batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))
    batch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    output_directory = fl.create_directory(f"OutputData/OutputFiles_Data_{sampleName}_{batch_time_suffix}", logger=None)

    # Initialize the batch logger
    log_file_path = os.path.join(output_directory, "BatchProcess.log")
    logger = setup_logger(log_file_path, logger_name="Batch")

    logger.info(f"Output Parent Directory Created: {output_directory}")
    logger.info(f"Batch started at: {batch_time}")

    logger.info(f"Loading calibrant file once...")
    try: 
        ai = pyFAI.load(config['poni_file'])
        q0_chi_ref = np.loadtxt(config['q0_reference_file'])
        logger.info("Calibrant and q0 reference files loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Either the poni or q0 reference file could not be loaded '{config['poni_file']}'. Aborting. Error: {e}")
        return

    tif_paths = sorted(glob.glob(os.path.join(input_dir, "*.tiff")))
    logger.info(f"Found {len(tif_paths)} .tif files in {input_dir}")

    def run_pipeline_for_file(tif_path, config, output_directory, ai, q0_chi_ref):
        # Initializing the per-image log files
        filename = os.path.splitext(os.path.basename(tif_path))[0]
        log_path = os.path.join(output_directory, f"{filename}_pipeline.log")
        file_logger = setup_logger(log_path, logger_name=filename)
        try:
            # Initialization of user-defined variables
            poni_file = config['poni_file']
            save_chi_files = config['save_chi_files']
            plot_q_vs_chi = config['plot_q_vs_chi']
            plot_strain_vs_chi = config['plot_strain_vs_chi']
            save_adjusted_tif = config['save_adjusted_tif']
            tif_file = tif_path
            mask_thresh = config['mask_thresh']
            num_azim_bins = config['num_azim_bins']
            q_min_nm1 = config['q_min_nm1']
            npt_rad = config['npt_rad']
            delta_tol = config['delta_tol']
            wavelength_nm = config['wavelength_nm']
            solved_strain_components = config['solved_strain_components']
            initial_q_guesses = config['initial_q_guesses']
            tol_array = np.array(config['tol_array']) # Ensure it's a numpy array
            eta0 = config['eta0']
            min_rsquared = config['min_rsquared']

            # Creation of the output directory
            filename_noext = fl.remove_filename_extension(tif_file)
            outputPath = os.path.join(output_directory, filename_noext)
            output_path = fl.create_directory(outputPath, logger=file_logger)

            data, mask = fl.load_and_prep_image( # loads and masks the TIFF for analysis
                tif_file,
                output_path=output_path,
                mask_threshold=mask_thresh,
                logger=file_logger,
                save_adjusted_tif=save_adjusted_tif
            )
            
            if save_chi_files:
                chi_path = fl.create_directory(f"{output_path}/ChiOutput", logger=file_logger) # output file directory creation for chi files
            else:
                chi_path = None

            # --- MAIN AZIMUTHAL INTEGRATION FUNCTION ----------------------------------
            I2d, q, chi = fl.integrate_2d( 
                ai, data, mask,
                num_azim_bins=num_azim_bins,
                q_min=q_min_nm1,
                npt_rad=npt_rad,
                output_dir=chi_path,
                save_chi_files=save_chi_files,
                logger=file_logger)
            
            # --- MAIN PEAK FITTING FUNCTION --------------------------------------------
            q_vs_chi, q_vs_chi_errors, q_chi_path = fl.fit_peaks_with_initial_guesses(
                I2d,
                q,
                initial_q_guesses,
                delta_tol=delta_tol,
                eta0=eta0,
                delta_array=tol_array,
                output_dir=output_path,
                logger=file_logger,
                n_jobs=1)
            if plot_q_vs_chi: # calling the q vs chi plotting function
                fl.plot_q_vs_chi_stacked(
                    file_path=q_chi_path,
                    output_dir=output_path,
                    dpi=600,
                    plot=True,
                    logger=file_logger)
                
            # --- MAIN LATTICE DISTORTION AND STRAIN CALCULATION FUNCTION -----------------
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
                logger=file_logger,
                min_rsquared=min_rsquared)
            if plot_strain_vs_chi: # calling the unfitted strain vs chi plotting function
                fl.plot_strain_vs_chi_stacked(
                    file_path=strain_vs_chi_file,
                    output_dir=output_path,
                    dpi=600,
                    plot=True,
                    logger=file_logger)
            # Write strain tensor components AND ERRORS to per-image JSON file
            strain_tensor_path = os.path.join(output_path, "strain_tensor.json")
            # The 'strain_params' is now a list of dictionaries, so we just save it directly
            with open(strain_tensor_path, 'w') as f:
                json.dump(strain_tensor_components, f, indent=2)
            file_logger.info(f"Successfully completed {tif_path}")

        except Exception as e: # error handling
            file_logger.error(f"Failed to process {tif_path}: {e}")

    # --- Runs the batch processing script in parallel ---
    Parallel(n_jobs=n_jobs)(
        delayed(run_pipeline_for_file)(tif_path, config, output_directory, ai, q0_chi_ref) for tif_path in tif_paths
    )

    logger.info(f"Proceeding to save strain tensor summary")
    # After processing, collect strain tensor results for summary
    strain_summary = []
    for tif_path in tif_paths:
        # Use the same filename_noext logic as in run_pipeline_for_file
        filename_noext = fl.remove_filename_extension(tif_path)
        output_path = os.path.join(output_directory, filename_noext)
        strain_tensor_path = os.path.join(output_path, "strain_tensor.json")
        if os.path.exists(strain_tensor_path):
            with open(strain_tensor_path, 'r') as f:
                tensor_data = json.load(f)
            strain_summary.append({
                "filename": filename_noext,
                "strain_tensor": tensor_data
            })
    summary_path = os.path.join(output_directory, "strain_tensor_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(strain_summary, f, indent=2)
    logger.info(f"Strain tensor summary saved to: {summary_path}")

    end_time = time.time()
    batch_runtime = end_time - start_time
    logger.info(f"Batch ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Batch Run Time: {batch_runtime/60} minutes")

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

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # spawn just defines a type of parallel processing
    config = { # --- main analysis configuration dictionary ---
        'input_dir': "InputFiles/25C_AO_inputs", # directory housing the input images
        'sampleName': "VB-APS-SSAO-6_25C", # name used to create output data files
        'poni_file': "calibration/Calibration_LaB6_100x100_3s_r8_mod2.poni", # calibration file path
        'q0_reference_file': "ValidationOutputFiles/VB-APS-SSAO-6_25C_Map-AO_000304/q0_vs_chi_FITTED.txt", # q0 reference file path
        'save_chi_files': False, # toggles saving the azimuthal q data for each bin
        'plot_q_vs_chi': False, # toggles plotting q vs chi plots
        'plot_strain_vs_chi': False, # toggles plotting unfitted strain vs chi plots
        'save_adjusted_tif': False, # toggles saving the adjusted TIF files
        'num_jobs_parallel': -2, # Uses all cores except for 1 to perform parallel calculations (-1 indicates using the maximum number of cores)
        'mask_thresh': 4e2, # mask threshold for pixels; not used unless chosen; can be left at 4e2
        'num_azim_bins': 120, # number of azimuthal bins for averaging the peaks
        'q_min_nm1': 14.0, # minimum q value for radial integration
        'npt_rad': 3000, # number of radial points from which to calculate peak centroids; ~2-3x radial pixel count
        'delta_tol': 0.1, # A tolerance value for finding peak centroids if tol_array is not filled in
        'wavelength_nm': 0.1729786687, # X-ray wavelength [nm]
        'solved_strain_components': 5, # 3=biaxial, 5=biaxial+shear, 6=full
        'initial_q_guesses': [ # Initial q-values retrieved from 1.FindingRefPeaks.py for the particular dataset in question
            17.961188,
            24.500613,
            26.267830,
            29.974002,
            35.926353,
            39.034769,
            44.513621,
            45.514461
        ],
        'tol_array': [ # The tolerance in q [nm^-1] for finding the peak centroids
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # looking up in q (larger values)
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # looking down in q (smaller values)
        ],
        'eta0': 0.5, # eta which is used for the Pseudo-Voigt; 0.5 works nicely
        'min_rsquared': 0 # set to 0 to not filter by R-squared
    }
    
    batch_main_pipeline(config) # calling the batch process