import numpy as np
import matplotlib.pyplot as plt
from pyFAI import load
from pyFAI.ext.bilinear import calc_cartesian_positions
from matplotlib.patches import Circle

# --- Load geometry from PONI file ---
poni_path = "calibration/Calibration_LaB6_100x100_3s_r8.poni"  # Replace with the path to your .poni file
ai = load(poni_path)

# --- Reflections (d-spacings in meters) ---
d_spacings = [2.04e-10, 1.43e-10, 1.23e-10, 1.00e-10]  # Example: multiple reflections

# --- Generate azimuthal angles (radians) ---
n_points = 1000
chi = np.linspace(0, 2 * np.pi, n_points)

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# --- Loop through each d-spacing and simulate ring ---
for d in d_spacings:
    theta = np.arcsin(ai.wavelength / (2 * d))
    two_theta = 2 * theta
    tth_array = np.full_like(chi, two_theta)
    chi_deg = np.rad2deg(chi)
    # Convert to pixel coordinates
    coords = calc_cartesian_positions(ai, tth_array, chi_deg)
    x_pix, y_pix = coords[:, 0], coords[:, 1]
    ax.plot(x_pix, y_pix, '.', markersize=1, label=f"d = {d*1e10:.2f} Å")

# --- Plot PONI and beam center ---
poni_x = ai.poni2 / ai.detector.pixel2
poni_y = ai.poni1 / ai.detector.pixel1
ax.plot(poni_x, poni_y, 'g+', markersize=12, label="PONI")

# Compute beam center using geometry (if rot1, rot2 = 0, it's same as PONI)
beam_x, beam_y = ai.get_beam_center(unit="pixel")
ax.plot(beam_x, beam_y, 'r+', markersize=12, label="Beam Center")

# --- Final plot formatting ---
ax.set_aspect("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title("Simulated Diffraction Rings")
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.show()