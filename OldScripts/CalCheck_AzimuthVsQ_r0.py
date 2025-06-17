#!/usr/bin/env python3
import numpy as np
import fabio,pyFAI
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyFAI import AzimuthalIntegrator, calibrant

# === User inputs ===
poni_file    = "Calibration_LaB6_100x100_3s_r8.poni"  # your .poni
image_file   = "calibration/ceria_lab6_exsitu_71p676keV_1145mm_100x100_3s_002265avg.tif"  # your 2D XRD image
bin_width    = 3.0    # degrees per azimuthal sector
npt_radial   = 2000   # number of q‐points per radial integration
dq_window    = 0.1    # search half‐width around each expected q_peak
unit         = "q_nm^-1"  # will give q in 2π/d (nm⁻¹)

# === 1. load integrator ===
ai = pyFAI.integrator.azimuthal.AzimuthalIntegrator.from_poni(poni_file)

# === 2. load image ===
img = fabio.open(image_file).data

# === 3. get reference LaB6 q‐values ===
lab6 = calibrant.LaB6()
d_spacings = lab6.get_d()            # in nanometers
q_ref      = 2 * np.pi / d_spacings  # q = 2π/d

# === 4. loop over azimuthal sectors ===
az_edges = np.arange(0, 360+bin_width, bin_width)
az_centers = (az_edges[:-1] + az_edges[1:]) / 2.0

# prepare storage: one list per ring
q_peaks = [ [] for _ in q_ref ]

for θ0, θ1 in zip(az_edges[:-1], az_edges[1:]):
    # radial integration over this sector
    q_axis, intensity = ai.integrate1d(
        img,
        npt_radial,
        unit=unit,
        azimuth_range=(θ0, θ1)
    )
    # for each ring, find the local maximum near q_ref[i]
    for i, q0 in enumerate(q_ref):
        mask = (q_axis > (q0 - dq_window)) & (q_axis < (q0 + dq_window))
        if not np.any(mask):
            q_peaks[i].append(np.nan)
            continue
        sub_I = intensity[mask]
        sub_q = q_axis[mask]
        # pick the highest point (simple)
        idx = np.argmax(sub_I)
        q_peaks[i].append(sub_q[idx])

# === 5. plot ===
fig, axes = plt.subplots(len(q_ref), 1, sharex=True, figsize=(6, 12))
for i, ax in enumerate(axes):
    ax.plot(az_centers, q_peaks[i], '-', lw=1)
    ax.set_ylabel(f"q₂π/d, ring {i+1}")
    ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel("Azimuth (°)")
fig.suptitle("Azimuthal Variation of LaB₆ Ring Q‑Positions")
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()
