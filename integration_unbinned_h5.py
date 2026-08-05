"""
    - Derived from Aaron's Script 1
    - Alternative Unbinned Integration with HDF5 Output
    - Integrates with full cake (360º) in batch for a series of map scans
    - Supports Use of Andrew's waxs_peakfit and waxs_viewer package
    - Targeting one run, yielding one .h5 per map scan for readability with Andrew's code
    - Cannibalizing some of Aaron's functions and trying to streamline

    - This version unlikely to yield individual data files for easy user viewing (although we could, but is this just clutter?)... 
    retain Aaron's original script for that
"""

import FunctionLibrary as fl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pyFAI
import fabio
import os
import h5py
from pyFAI.io import HDF5Writer


################# OVERARCHING SAMPLE NOTES ##########################

#Sample 6: February Dataset
#Samples 5, 3, 2: October Dataset

########### CALIBRATION PARAMETERS #####################

#FEBRUARY 2025 BEAMTIME CALIBRATION PARAMETERS (BEN): 
poni_file = "0_calibration/Calibration_Feb25_ceria_1145mm_25C_att000_000112.poni" # calibration PONI file. Ben used CeO2
detector_type = "GE" # "Pilatus" or "GE"
mask_file = None
visit = "Feb2025" #Subfolder to separate full cake results by visit.
#Note: local_folder = "/Users/benjaminschneiderman/APS_Data_Local/APS_2025-02/InputFiles"

#OCTOBER 2025 BEAMTIME CALIBRATION PARAMETERS: 
# poni_file = "0_calibration/Calibration_Oct25_ceria_900mm_linkam_30C_att000_0006091.poni" # calibration PONI file. I used CeO2 
# detector_type = "Pilatus" # "Pilatus" or "GE"
# mask_file = "0_calibration/pilatus_mask.msk" # Either "path/to/your/mask.tif" or None
# visit = "Oct2025"
# Note: local_folder = "/Users/benjaminschneiderman/APS_Data_Local/APS_2025-10/pilatus"

########################################################

#Parameters for the map scan you will package into a single hdf5
local_folder = "/Users/benjaminschneiderman/APS_Data_Local/APS_2025-02/InputFiles" #Point to local storage to avoid cluttering OneDrive
isolated_mapscan_location = "Feb2025_OnHeat_25C" #Grouping the maps for organization
beamtime_given_prefix = "VB-APS-SSAO-6_25C_TestMap-AO_"
scan_range = (169, 520) #Beamtime assigned scan IDs

# Reference path: InputFiles/Feb2025_OnHeat_25C/VB-APS-SSAO-6_25C_TestMap-AO_000169.avg.tiff

def get_tif_file(scan_id):
    return os.path.join(local_folder, isolated_mapscan_location, f"{beamtime_given_prefix}{scan_id:06d}.avg.tiff")


def main():

    # Create output location for the single .h5, outside the main loop
    output_path = os.path.join("1_UnbinnedIntegration_PeakFinding", visit, isolated_mapscan_location)
    fl.create_directory(output_path, exist_ok=True)

    # Intialize .hdf5 writer: single file per map scan
    h5_path = os.path.join(output_path, f"scan_range_{scan_range[0]}_{scan_range[1]}.h5")
    writer = HDF5Writer(filename=h5_path)
    writer.init_file()

    # Inclusive range from scan_range[0] to scan_range[1]
    for scan_id in range(scan_range[0], scan_range[1] + 1):
        indiv_file_path = get_tif_file(scan_id)

        # Skip missing frame files gracefully if a scan in the range was aborted/missing
        if not os.path.exists(indiv_file_path):
            print(f"Warning: File not found for scan {scan_id:06d}, skipping...")
            continue

        # Load image data
        image = fabio.open(indiv_file_path).data
        # EXTREMELY IMPORTANT: Flip the image AND MASK vertically for Pilatus detector: MATCHES Oct. 25 CALIBRATION
        image = np.flipud(image) if detector_type == "Pilatus" else image   

        # Perform 1D pyFAI integration
        npt = 2000 # number of radial bins
        result = ai.integrate1d(image, npt, mask=mask, dummy=np.nan, unit="q_nm^-1")

        # Format entry_name using zero-padded scan_id to keep HDF5 keys sorted naturally
        entry_key = f"scan_{scan_id:06d}"

        # Append integration result into the open .h5 file
        writer.write(result, entry_name=entry_key)

    # Close the HDF5 writer once all scans are processed
    writer.close()
    print(f"UNBINNED INTEGRATION: Successfully saved scans {scan_range[0]} through {scan_range[1]} into: {h5_path}")


    #######AARON Script 1########

    # This removes the file extension and .avg from the end of the averaged image files
    filename = fl.remove_filename_extension(tif_file)
    
    # Creates an output directory of the same name as the mapping image to store all the data for that map image location
    outputPath = os.path.join("1_UnbinnedIntegration_PeakFinding", visit, descriptor, filename)
    output_path = fl.create_directory(outputPath)
    print(f"[INFO] Output Path is {output_path}")

    # Load the pyFAI integrator from the .poni calibration file
    ai = pyFAI.load(poni_file)

    # Load image data from .tif
    image = fabio.open(tif_file).data
    # EXTREMELY IMPORTANT: Flip the image AND MASK vertically for Pilatus detector: MATCHES Oct. 25 CALIBRATION
    image = np.flipud(image) if detector_type == "Pilatus" else image   

    # Load mask if provided
    if mask_file:
        mask = fabio.open(mask_file).data == 1  # .astype(bool)
        mask = np.flipud(mask) if detector_type == "Pilatus" else mask # EXTREMELY IMPORTANT: Flip the MASK vertically
        fl.print_mask(mask, "0_calibration/pilatus_mask.tif") #Check to make sure mask is correct (e.g., the orientation)
    else:
        mask = None

    # Perform azimuthal integration to get 1D pattern (q vs I)
    npt = 2000 # number of radial bins
    writer = HDF5Writer(filename=f"{output_path}/unbinned_integration.h5", h5path="entry/pyFAI")
    writer.init_file()
    result = ai.integrate1d(image, npt, mask=mask, dummy=np.nan, unit="q_nm^-1")
    writer.write(result, entry_name=filename)
    writer.close()
    q = result.radial
    I = result.intensity

    # Save temporary file to use with validate_curve_fitting()
    # temp_int_file = "temp_intensity.int"
    # np.savetxt(temp_int_file, np.column_stack((q, I, np.zeros_like(q))), fmt="%.6f")

    # Call validate_curve_fitting to fit peaks
    peak_positions_q = fl.fit_peak_centroids(q, I, height_frac=height_frac, distance=distance)
    peak_positions_d = 2 * np.pi / np.array(peak_positions_q)
    # os.remove(temp_int_file)

    # Save peaks to file
    output_txt = os.path.join(output_path,"peak_positions.txt")
    np.savetxt(output_txt, np.column_stack((peak_positions_q, peak_positions_d)),
                 fmt="%.6f", delimiter="\t", header="q [nm^-1] \t d [nm]")
    print(f"Detected {len(peak_positions_q)} peaks. Saved to {output_txt}")

    # Save q & I data to a file
    np.savetxt(f"{output_path}/q_vs_I.txt", np.column_stack((q, I)), fmt="%.6f", delimiter=" ")

    # Plot the pattern and detected peaks
    plt.figure(figsize=(5, 3))
    plt.plot(q, I, label='Integrated pattern', linewidth=1.0, color='k')
    plt.plot(peak_positions_q, [np.interp(p, q, I) for p in peak_positions_q], 'rx', label='Fitted Peaks')
    plt.xlabel("q [nm$^{-1}$]")
    plt.ylabel("Intensity [a.u.]")
    # plt.title("1D Azimuthally Integrated Pattern with Peak Locations")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.set_xlim(10,90)
    
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/peak_detection_plot.png", dpi=600)
    # plt.show()

if __name__ == "__main__":
    main()