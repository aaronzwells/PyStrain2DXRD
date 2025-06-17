import pyFAI, fabio
import numpy as np
from pyFAI.gui import jupyter
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

# Print PyFAI version for tracking purposes
print("\n\nPyFAI version:", pyFAI.version)

# 1) Load predetermined calibration and initialize Azimuthal Integrator (ai) object
ai = pyFAI.load("calibration/Calibration_LaB6_100x100_3s_r0.poni")
print("\nIntegrator: \n", ai)

# 2) Load diffraction image for binning
img = fabio.open("InputFiles/AVG_AO_Ref_100x100_AvgStack.tif")
print("Image:", img)

img_array = img.data
print("img_array:", type(img_array), img_array.shape, img_array.dtype)
mask = img_array>4e9

res2 = ai.integrate2d_ng(img_array,
                         1000, 120,
                         unit="q_nm^-1",
                         filename="integrated.edf")

cake = fabio.open("integrated.edf")
print(cake.header)
print("cake:", type(cake.data), cake.data.shape, cake.data.dtype)
