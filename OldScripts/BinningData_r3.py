import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pyFAI, fabio

# 1) load integrator and image
ai = pyFAI.load("calibration/Calibration_LaB6_100x100_3s_r4.poni")
img = fabio.open("InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif")
data = img.data
mask = data > 4e9

# 2) do a 2D integration and grab the axes
# quick full radial‐axis grab
_q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial

res2d = ai.integrate2d(
    data,
    npt_rad=5000,
    npt_azim=120,
    unit="q_nm^-1",
    mask=mask,
    radial_range=(16.0, _q_full[-1]),  # drop anything q < 12
)
I2d   = res2d.intensity    # 2D array of shape (npt_azim, npt_rad)
q     = res2d.radial       # radial (q) bin centres, length npt_rad
chi   = res2d.azimuthal    # azimuth (χ) bin centres, length npt_azim :contentReference[oaicite:0]{index=0}

# if intensity comes back as (radial,azim), transpose:
if I2d.shape == (len(q), len(chi)):
    I2d = I2d.T

# 3) find the global peak positions on the azimuthal average
radial_mean = I2d.mean(axis=0)
peaks, _   = find_peaks(radial_mean, height=np.max(radial_mean)*0.09, distance=20)
q_peaks    = q[peaks]
# pick the top N peaks (e.g. 8), sorted by q
q_peaks    = np.sort(q_peaks)[:8]

# 4) for each χ‑slice, locate the local peak nearest each global q
n_peaks = len(q_peaks)
q_vs_chi = np.zeros((n_peaks, len(chi)))

noise_tol = 0.05  # nm⁻¹ maximum allowed deviation from the mean

for i, intensity_row in enumerate(I2d):
    local_inds, _ = find_peaks(intensity_row,
                               height=np.max(intensity_row)*0.03)
    local_qs = q[local_inds]
    for j, q0 in enumerate(q_peaks):
        # find the closest local peak
        if local_qs.size>0:
            idx = np.argmin(np.abs(local_qs - q0))
            qval = local_qs[idx]
            # only accept it if within ±noise_tol of the global q0
            if abs(qval - q0) <= noise_tol:
                q_vs_chi[j, i] = qval
            else:
                q_vs_chi[j, i] = np.nan   # or q0, or leave as zero
        else:
            q_vs_chi[j, i] = np.nan

# 5) plot in stacked subplots
fig, axes = plt.subplots(n_peaks, 1, sharex=True, figsize=(6, 2*n_peaks))
for j, ax in enumerate(axes):
    ax.plot(chi, q_vs_chi[j], "-k", lw=1)
    ax.set_ylabel(f"q ≃ {q_peaks[j]:.2f}")
    ax.set_ylim(np.nanmean(q_vs_chi[j])-0.05, np.nanmean(q_vs_chi[j])+0.05)
axes[-1].set_xlabel("Azimuth (°)")
fig.tight_layout()
plt.show()

