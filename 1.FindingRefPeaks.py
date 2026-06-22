"""Integrate 2D diffraction patterns and find peaks for a single image"""

import FunctionLibrary as fl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pyFAI
import fabio
import os

######## SCRIPT 1 PARAMETERS ########

#Sample 6: February Dataset
#Samples 5, 3, 2: October Dataset

#OCTOBER 2025 BEAMTIME CALIBRATION PARAMETERS: 
poni_file = "calibration/Calibration_Oct25_ceria_900mm_linkam_30C_att000_0006091.poni" # calibration PONI file. I used CeO2 
detector_type = "Pilatus" # "Pilatus" or "GE"
mask_file = "calibration/pilatus_mask.msk" # Either "path/to/your/mask.tif" or None

#FEBRUARY 2025 BEAMTIME CALIBRATION PARAMETERS: (This calibration gives incorrect CeO2 rm. temp values! But is what Aaron used.)
# poni_file = "calibration/Calibration_LaB6_100x100_3s_r8_mod2.poni" # calibration PONI file. Aaron used LaB6
# detector_type = "GE" # "Pilatus" or "GE"
# mask_file = None

#PLUG IN SINGLE IMAGE TO INGEGRATE AND FIND PEAKS
#2026 Analysis uses include: Linkam temperature calibration (a few files, not looped), checking Feb/Oct peak position discrepancy
tif_file = "InputFiles/Oct2025_linkam_temperature_calib/ceria_900mm_linkam_30C_att000/ceria_900mm_linkam_30C_att000_0006092.tif" # representative data TIF file


def main(
        poni_file=poni_file, 
        tif_file=tif_file, 
        height_frac=0.1, 
        distance=20):
    
    # This removes the file extension and .avg from the end of the averaged image files
    filename = fl.remove_filename_extension(tif_file)
    
    # Creates an output directory of the same name as the mapping image to store all the data for that map image location
    outputPath = os.path.join("PeakFinding", filename)
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
        fl.print_mask(mask, "calibration/pilatus_mask.tif") #Check to make sure mask is correct (e.g., the orientation)
    else:
        mask = None

    # Perform azimuthal integration to get 1D pattern (q vs I)
    npt = 2000 # number of radial bins
    result = ai.integrate1d(image, npt, mask=mask, dummy=np.nan, unit="q_nm^-1")
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