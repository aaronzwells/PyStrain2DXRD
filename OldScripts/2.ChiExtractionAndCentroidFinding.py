import FunctionLibrary as fl
import time
import os
import numpy as np

# --- Main pipeline -------------------------------------------------------
def main(tif_override=None, output_tensor_path=None):
    start_time = time.time()
    os.makedirs("OutputFiles/Fits", exist_ok=True)
    poni_file     = "calibration/Calibration_LaB6_100x100_3s_r8.poni"
    tif_file      = tif_override or "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg.tif"
    # ref_tif_path  = "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif"
    mask_thresh   = 4e2 # threshold value for the image mask
    num_azim_bins = 120 # number of azimuthal bins around the data
    q_min_nm1     = 16.0 # q_0 for binning of the data
    npt_rad       = 5000 # number of radial bins
    wavelength    = 0.1729786687 # X-Ray wavelength in nm
    delta_tol     = 0.02
    initial_q_guesses = [17.96, 24.50, 26.27, 29.97, 35.92, 39.02, 44.50, 45.51]
    delta_array   = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], # larger q
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]) # smaller q
    eta0          = 0.5
    harmonics     = (1, 2)
    qazi_png       = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_fourier_{num_azim_bins}bin.png"
    radar_png      = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_radar_{num_azim_bins}bin.png"

    q_peak_locs = fl.convert_2theta_to_q(
        "AdditionalFiles/FxnValidation/FitPeakLocations-Al2O3.txt", 
        wavelength_nm=wavelength)               
    print(q_peak_locs)
    ai, data, mask = fl.load_integrator_and_data(
        poni_file,
        tif_file,
        mask_threshold=mask_thresh)
    I2d, q, chi = fl.integrate_2d(
        ai, data, mask,
        num_azim_bins=num_azim_bins,
        q_min=q_min_nm1,
        npt_rad=npt_rad,
        output_dir="OutputFiles/ChiOutput"
    )
    q_vs_chi = fl.fit_peaks_with_initial_guesses(I2d, q, initial_q_guesses,
                                                delta_tol=delta_tol, 
                                                eta0=eta0, delta_array=delta_array)
fl.plot_q_vs_chi_stacked(
    file_path="OutputFiles/QvsChi_data/q_vs_chi_peaks.txt",
    dpi=600)

if __name__ == "__main__":
    main()