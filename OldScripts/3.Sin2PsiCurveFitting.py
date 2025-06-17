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
    mask_thresh   = 4e2
    num_azim_bins = 120
    q_min_nm1     = 16.0
    npt_rad       = 5000
    delta_tol     = 0.07
    eta0          = 0.5
    harmonics     = (1, 2)
    qazi_png       = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_fourier_{num_azim_bins}bin.png"
    radar_png      = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_radar_{num_azim_bins}bin.png"

    fl.fit_lattice_cone_distortion("OutputFiles/QvsChi_data/q_vs_chi_peaks.txt")

if __name__ == "__main__":
    main()
