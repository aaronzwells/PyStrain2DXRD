import logging
import FunctionLibrary as fl
import time
import os
import numpy as np
from joblib import Parallel, delayed
import glob
import json

# Batch processing for all .tif files in InputFiles/AOInputs
def batch_main_pipeline(input_dir="InputFiles/400C_AO_inputs", n_jobs=-1):
    sampleName = "VB-APS-SSAO-6_400C"
    start_time = time.time()
    batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))
    batch_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    output_directory = fl.create_directory(f"OutputData/OutputFiles_Data_{sampleName}_{batch_time_suffix}", logger=None)

    log_file_path = os.path.join(output_directory, "BatchProcess.log")
    logger = setup_logger(log_file_path, logger_name="Batch")

    logger.info(f"Output Parent Directory Created: {output_directory}")
    logger.info(f"Batch started at: {batch_time}")

    tif_paths = sorted(glob.glob(os.path.join(input_dir, "*.tiff")))
    logger.info(f"Found {len(tif_paths)} .tif files in {input_dir}")

    def run_pipeline_for_file(tif_path):
        filename = os.path.splitext(os.path.basename(tif_path))[0]
        log_path = os.path.join(output_directory, f"{filename}_pipeline.log")
        file_logger = setup_logger(log_path, logger_name=filename)
        try:
            file_logger.info(f"Starting pipeline for {tif_path}")
            poni_file     = "calibration/Calibration_LaB6_100x100_3s_r8_mod2.poni"
            tif_file      = tif_path
            mask_thresh   = 4e2
            num_azim_bins = 120
            q_min_nm1     = 14.0
            npt_rad       = 3000
            delta_tol     = 0.1
            wavelength_nm = 0.1729786687 # [nm] X-ray wavelength
            solved_strain_components = 5 # 3 = biaxial; 5 = biaxial w/ shear; 6 = all components
            initial_q_guesses = [
                17.911220,
                24.433947,
                26.201850,
                29.895184,
                35.831000,
                38.928253,
                44.393175,
                45.392588
            ]
            tol_array   = np.array([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
            eta0          = 0.5


            filename_noext = fl.remove_filename_extension(tif_file)
            outputPath = os.path.join(output_directory, filename_noext)
            output_path = fl.create_directory(outputPath, logger=file_logger)
            ai, data, mask = fl.load_integrator_and_data(
                poni_file,
                tif_file,
                output_path=output_path,
                mask_threshold=mask_thresh,
                logger=file_logger)
            chi_path = fl.create_directory(f"{output_path}/ChiOutput", logger=file_logger)
            I2d, q, chi = fl.integrate_2d(
                ai, data, mask,
                num_azim_bins=num_azim_bins,
                q_min=q_min_nm1,
                npt_rad=npt_rad,
                output_dir=chi_path,
                logger=file_logger)
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
                plot=False,
                logger=file_logger)
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
            fl.plot_strain_vs_chi_stacked(
                file_path=strain_vs_chi_file,
                output_dir=output_path,
                dpi=600,
                plot=True,
                logger=file_logger)
            # Write strain tensor components to per-image JSON file
            strain_tensor_path = os.path.join(output_path, "strain_tensor.json")
            tensor_data = [
                {
                    "ring": i + 1,
                    "eps_xx": row[0],
                    "eps_xy": row[1],
                    "eps_yy": row[2],
                    "eps_xz": row[3],
                    "eps_yz": row[4],
                    "eps_zz": row[5]
                }
                for i, row in enumerate(strain_tensor_components.tolist())
            ]
            with open(strain_tensor_path, 'w') as f:
                json.dump(tensor_data, f, indent=2)
            file_logger.info(f"Successfully completed {tif_path}")
        except Exception as e:
            file_logger.error(f"Failed to process {tif_path}: {e}")

    Parallel(n_jobs=n_jobs)(
        delayed(run_pipeline_for_file)(tif_path) for tif_path in tif_paths
    )

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
    batch_main_pipeline()