import FunctionLibrary_r1 as fl
import time
import os
import numpy as np

# Running the validation function to ensure the Pseudo-Voigt curve fitting function is correct
centroids = fl.validate_curve_fitting("AdditionalFiles/FxnValidation/IdealIntensityPlot-Al2O3.int")
print("Centroid positions (2θ): ", centroids)
