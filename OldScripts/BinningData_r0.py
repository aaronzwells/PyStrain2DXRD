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
q_bins = 1000 # number of Q bins

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

# 4) Prepare storage for peak locations and widths
num_peaks = 8
num_bins, q_bins = all_q.shape
peak_positions = np.full((num_bins, num_peaks), np.nan)
peak_fwhm      = np.full((num_bins, num_peaks), np.nan)

# Define a minimal relative height threshold so you don't pick noise:
# here 10% of the max intensity in each slice
rel_height = 0.5  # for FWHM
for i in range(num_bins):
    intens = all_I[i]
    q_ax   = all_q[i]

    # a) detect peaks above 10% of the max
    thresh = np.max(intens) * 0.03
    peaks, props = find_peaks(intens, height=thresh)

    if len(peaks) == 0:
        continue

    # b) compute widths @ half max for those peaks
    results = peak_widths(intens, peaks, rel_height=rel_height)
    # results[2] and [3] are the left/right interpolated indices

    # c) take the first num_peaks by ascending q position
    #    sort peaks by their q value
    order = np.argsort(q_ax[peaks])[:num_peaks]
    sel_peaks = peaks[order]
    sel_left  = results[2][order]
    sel_right = results[3][order]

    # d) record the peak q and FWHM
    peak_positions[i, :len(sel_peaks)] = q_ax[sel_peaks]
    peak_fwhm[i, :len(sel_peaks)]      = q_ax[sel_right.astype(int)] \
                                        - q_ax[sel_left.astype(int)]

num_bins, num_peaks = peak_positions.shape

# x values: bin indices 0,1,2,…,num_bins-1
bins = np.arange(num_bins)

# create one subplot per peak
fig, axes = plt.subplots(num_peaks, 1, sharex=True, figsize=(6, 0.9*num_peaks))
for j in range(num_peaks):
    ax = axes[j]
    ax.plot(bins, peak_positions[:, j], linestyle='-')
    ax.set_ylabel(f"Q [nm^-1]")
    ax.grid(linestyle=':', alpha=0.5)
    ax.set_ylim(np.mean(peak_positions[:,j])-.05,np.average(peak_positions[:,j])+.05)

axes[-1].set_xlabel("Azimuthal Bin Index")
plt.tight_layout()
plt.show()