import pyFAI, fabio
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

# Print PyFAI version for tracking purposes
print("\n\nPyFAI version:", pyFAI.version)

# 1) Load predetermined calibration
ai = pyFAI.load("calibration/Calibration_LaB6_100x100_3s_r0.poni")
print("\nIntegrator: \n", ai)

# 2) Load diffraction image for binning
img = fabio.open("InputFiles/AVG_AO_Ref_100x100_AvgStack.tif")
print("Image:", img)

img_array = img.data
print("img_array:", type(img_array), img_array.shape, img_array.dtype)
mask = img_array>4e9

# 3) Looping over bins to retrieve all azimuth vs Q data
#   3.1) Define the bins (azimuthal and radial)
#       3.1.1) azimuthal bin definitions
bin_width_deg = 3.0                       # degrees
bin_width = bin_width_deg*np.pi/180
num_bins = int(360/bin_width)
bin_start =  0.0                      # e.g. the 0–3° bin
bin_end   = bin_start + bin_width
#       3.1.2) radial bin definitions
q_bins = 2000 # number of Q bins

all_q = np.empty((num_bins,q_bins),dtype=float)
all_I= np.empty((num_bins,q_bins),dtype=float)

# Initializing the loop
for i in range(int(num_bins)):
    start = i*bin_width
    end = start + bin_width
#   3.2) Perform 1D integration on a single azimuthal bin
    q, I = ai.integrate1d_ng(img_array,
                            npt=q_bins,
                            unit="q_nm^-1",
                            azimuth_range = (bin_start, bin_end))
    all_q[i,:] = q
    all_I[i,:] = I

#       3.2.1) (optional) Inspecting results in the print readout
# print(all_q[5])
# print(all_I[5])

# === 4) Peak‐center finding per azimuthal bin ===
# Use the same initial guesses as in Pk_Fit_ExpData.m:
p0_cen   = [18, 24.5, 26.2, 30, 36, 39, 44.5, 45.5]  # initial Q guesses
p0_fwhm  = [0.05] * len(p0_cen)                     # not used here
p0_dup   = [10, 10,  5, 10, 10, 10, 10, 10]          # up‐search window
p0_ddown = [10, 10,  3, 10, 10, 10, 10, 10]          # down‐search window

num_peaks     = len(p0_cen)
num_bins, _   = all_q.shape
peak_centers  = np.full((num_bins, num_peaks), np.nan)

for i in range(num_bins):
    q_line = all_q[i]
    I_line = all_I[i]

    for j in range(num_peaks):
        cen0  = p0_cen[j]
        lower = cen0 - p0_ddown[j]
        upper = cen0 + p0_dup[j]
        # restrict to the local window around the expected peak
        mask  = (q_line >= lower) & (q_line <= upper)
        if not np.any(mask):
            continue

        q_win = q_line[mask]
        I_win = I_line[mask]

        # compute a center‐of‐mass centroid as the “peak center”
        peak_centers[i, j] = np.sum(q_win * I_win) / np.sum(I_win)

# === 5) Plot exactly as the MATLAB script ===
# x = bin centers in degrees
az_centers = (np.arange(num_bins) + 1) * (360/num_bins)

fig, axes = plt.subplots(num_peaks, 1, sharex=True, figsize=(6, 2*num_peaks))
for j in range(num_peaks):
    ax = axes[j]
    ax.plot(az_centers, peak_centers[:, j], 'k-', lw=1)
    ax.set_ylabel(f'q = 2π/d, peak {j+1}')
    # set y‐limits to mean±0.05, as in the MATLAB code
    m = np.nanmean(peak_centers[:, j])
    ax.set_ylim(m-0.05, m+0.05)
    ax.grid(linestyle=':', alpha=0.5)

axes[-1].set_xlabel('Azimuth (°)')
plt.tight_layout()
plt.show()
