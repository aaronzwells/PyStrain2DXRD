import FunctionLibrary as fl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pyFAI
import fabio
import os

poni_file = "calibration/Calibration_LaB6_100x100_3s_r8_mod2.poni" # calibration PONI file
tif_file = "InputFiles/VB-APS-SSAO-6_25C_TestMap-AO_000288.avg.tiff" # representative data TIF file

def main(
        poni_file=poni_file, 
        tif_file=tif_file, 
        height_frac=0.05, 
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

    # Perform azimuthal integration to get 1D pattern (q vs I)
    npt = 2000
    result = ai.integrate1d(image, npt, unit="q_nm^-1")
    q = result.radial
    I = result.intensity

    # Save temporary file to use with validate_curve_fitting()
    # temp_int_file = "temp_intensity.int"
    # np.savetxt(temp_int_file, np.column_stack((q, I, np.zeros_like(q))), fmt="%.6f")

    # Call validate_curve_fitting to fit peaks
    peak_positions_q = fl.fit_peak_centroids(q, I, height_frac=height_frac, distance=distance)
    # os.remove(temp_int_file)

    # Save peaks to file
    output_txt = os.path.join(output_path,"peak_positions.txt")
    np.savetxt(output_txt, peak_positions_q, fmt="%.6f", header="q positions of detected peaks [nm^-1]")
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