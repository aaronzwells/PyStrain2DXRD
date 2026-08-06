import FunctionLibrary_r1 as fl
import numpy as np
import matplotlib.pyplot as plt
import time

strain_path = "OutputFiles_Data_VB-APS-SSAO-6_25C_2025.08.19-15.41.01/strain_tensor_summary.json" # path to the json housing the strain data
wavelength = 0.1729786687 # [nm] X-ray wavelength
num_azim_bins = 120 # number of azimuthal bins
sampleName = "VB-APS-SSAO-6_25C"

#Creating Ouput directory based upon run start time
start_time = time.time()
batch_time_suffix = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime(start_time))
output_directory = fl.create_directory(f"ValidationOutputFiles/ReconstructedData_{sampleName}", logger=None)

fl.reconstruct_rings_from_json(
    json_path=strain_path, 
    wavelength_nm=wavelength, 
    chi_step=360/num_azim_bins, 
    logger=None, 
    plot=True, 
    output_dir=output_directory
    )