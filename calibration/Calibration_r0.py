#!/usr/bin/env python3
"""
A sample script to calibrate a detector using a LaB6 calibrant image
with pyFAI, producing a calibration (poni) file.

Required libraries:
  • pyFAI
  • fabio
  • numpy

You can install these via pip if necessary:
    pip install pyfai fabio numpy
"""

import fabio
import numpy as np
import pyFAI
from pyFAI import calibrant

# ---------------------------
# 1. User-Provided Inputs
# ---------------------------
# Path to the LaB6 calibration image (e.g., in TIFF format)
image_file = "InputFiles/ceria_lab6_exsitu_71p676keV_1145mm_100x100_3s_002265avg.tif"  # change to your file path

# Incident X-ray wavelength in meters (for example, 1.54 Å corresponds to 1.54e-10 m)
wavelength = 0.17298e-10

# Initial guess for the sample-to-detector distance in meters (adjust as needed)
distance_guess = 1.145

# Detector pixel size in meters (make sure to set this correctly; e.g., 100 um = 100e-6 m)
pixel_size = 200e-6

# Initial guess for the beam center (poni1, poni2) in pixel coordinates
# These numbers can be approximated from the diffraction image.
center_guess = (1024, 1024)  # change based on your detector geometry

# (Optional) Detector tilt angles or other parameters can be provided if known.
# For a flat detector with no tilt, these are not needed.

# ---------------------------
# 2. Read the Calibration Image
# ---------------------------
# Using fabio to read the image file into a numpy array
img = fabio.open(image_file).data

# ---------------------------
# 3. Set Up the Azimuthal Integrator
# ---------------------------
# Create an AzimuthalIntegrator instance with the initial guess parameters.
# The detector type here is specified as "FlatPanel", which is common;
# adjust if you have a different detector.
ai = pyFAI.AzimuthalIntegrator(
    dist=distance_guess,
    pixel1=pixel_size,
    pixel2=pixel_size,
    poni1=center_guess[0],
    poni2=center_guess[1],
    wavelength=wavelength
)

# ---------------------------
# 4. Perform the Calibration
# ---------------------------
# Use the built-in LaB6 calibrant information from pyFAI.
# The calibrant object contains the known d-spacings and diffraction ring positions.
#
# The calibrate2 method uses ring positions in the input image to refine:
# • Detector distance
# • Beam center (poni1, poni2)
# • (Optionally) additional parameters (e.g., tilt) if provided
# You can also specify a 2θ range to focus on a subset of diffraction rings.
#
# Here, we perform the calibration with interactive refinement turned off.
ai.calibrate2(img, calibrant=calibrant.LaB6(), twoTheta_range=[10, 50])

# ---------------------------
# 5. Save the Calibration File
# ---------------------------
# Save the refined parameters into a calibration file (poni file) that can be used
# with pyFAI integration routines.
calibration_filename = "calibration.poni"
ai.save(calibration_filename)
print(f"Calibration file saved as {calibration_filename}")
