import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter
import pyFAI
from pyFAI.calibrant import get_calibrant
from pyFAI import load

# === 1. Load your calibrated integrator from .poni file ===
poni_path = "calibration/Calibration_LaB6_100x100_3s_r8.poni"  # <- Update this line with your file path
ai = load(poni_path)
detector = ai.detector
shape = detector.shape
pixel_size = detector.pixel1  # assume square pixels

# === 2. Define d-spacings manually (in nanometers) ===
d_spacings_nm = np.array([
    0.348,  # (0 1 2)
    0.2576, # (1 0 -1 4)
    0.2403, # (2 -1 -1 0)
    0.2106, # (2 -1 -1 3) 
    # 0.238,  # (104)
    # 0.207,  # (110)
    0.1757, # (2 0 -2 4)
    # 0.167,  # (113)
    0.1617, # (2 -1 -1 6)
    0.1525, # (1 0 -1 8)
    # 0.146,  # (024)
    0.1418, # (3 -1 -2 8)
    0.1387, # (3 0 -3 0)
    # 0.132,  # (116)
    0.1246, # (2 -1 -1 9)
    0.1201, # (4 -2 -2 0)
    # 0.118,  # (214)
    0.1158, # (4 -2 -2 3)
    0.1135, # (3 -1 -2 8)
    0.111,  # (2 0 -2 10)
    0.1089, # (4 -1 -3 4)
    0.1053, # (4 -2 -2 6)
])  # These are d-spacings retrieved from https://next-gen.materialsproject.org/materials/mp-1143

# === Convert to 2θ using Bragg's law: 2θ = 2 * arcsin(λ / (2d)) ===
wavelength_nm = ai.wavelength * 1e9  # convert wavelength from m to nm
two_thetas = 2 * np.degrees(np.arcsin(wavelength_nm / (2 * d_spacings_nm)))

# === 3. Simulate 2D diffraction pattern with rings ===
image = np.zeros(shape)

for angle in two_thetas:
    # Compute radial distance (in meters, then pixels)
    r_meters = ai.dist * np.tan(np.radians(angle))
    r_pixels = r_meters / pixel_size

    # Find detector center in pixels
    center_y = ai.poni1 / pixel_size
    center_x = ai.poni2 / pixel_size

    # Draw a ring (approximate as a thick disk)
    rr, cc = circle_perimeter(int(center_y), int(center_x), int(r_pixels), shape=shape)
    image[rr, cc] += 1  # Add intensity to ring

# === 4. Plot the simulated image ===
# === 4. Plot the simulated image ===
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='hot', origin='lower')
# plt.title("Simulated 2D Diffraction Pattern: α-Al₂O₃")
# plt.xlabel("Pixels")
# plt.ylabel("Pixels")
# plt.colorbar(label="Intensity (arb. units)")
plt.axis('off')
plt.savefig("simulated_pattern.tif", format='tiff', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.close()
plt.show()