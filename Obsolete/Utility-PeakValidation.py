# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, PolynomialModel

# Generate synthetic data with equal step size in two-theta
x = np.linspace(20, 80, 1000)  # Equal step size in two-theta values
true_peak = 40  # True peak position
true_width = 2  # True peak width
true_amplitude = 100  # True peak intensity
noise = np.random.normal(0, 10, size=x.shape)  # Add noise

# Add a linear background (N=1 polynomial)
background_slope = 0  # Slope of the linear background
background_intercept = 10  # Intercept of the linear background
background = background_slope * x + background_intercept

# Combine peak, background, and noise
y = true_amplitude * np.exp(-((x - true_peak) ** 2) / (2 * true_width ** 2)) + background + noise

# Define a Gaussian model for the peak
gaussian_model = GaussianModel()

# Define a polynomial model for the background (N=1 for linear background)
polynomial_model = PolynomialModel(degree=1)

# Combine the models
combined_model = gaussian_model + polynomial_model

# Create parameters for fitting
params = combined_model.make_params(
    center=38, sigma=2, amplitude=100,  # Gaussian parameters
    c0=10, c1=0.5  # Polynomial parameters (c0: intercept, c1: slope)
)

# Perform the fit
result = combined_model.fit(y, params, x=x)

# Extract fitting results
peak_position = result.params['center'].value
peak_height = result.params['height'].value
fwhm = result.params['fwhm'].value
r_squared = result.rsquared  # Extract R-squared value
background_intercept_fit = result.params['c0'].value
background_slope_fit = result.params['c1'].value

# Print fit report
print(result.fit_report())

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Data", color="blue", alpha=0.7)
plt.plot(x, result.best_fit, label="Fit", color="red", linewidth=2)

# Annotate fitting results on the plot
plt.text(peak_position + 5, peak_height - 10, f"Peak Position: {peak_position:.2f}", color="red", fontsize=10)
plt.text(peak_position + 5, peak_height - 18, f"Peak Height: {peak_height:.2f}", color="red", fontsize=10)
plt.text(peak_position + 5, peak_height - 26, f"FWHM: {fwhm:.2f}", color="red", fontsize=10)
plt.text(peak_position + 5, peak_height - 34, f"R-squared: {r_squared:.4f}", color="green", fontsize=10)
plt.text(peak_position + 5, peak_height - 42, f"Background: y = {background_slope_fit:.2f}x + {background_intercept_fit:.2f}", color="purple", fontsize=10)

# Add labels and legend
plt.legend()
plt.xlabel("Two-theta")
plt.ylabel("Intensity")
plt.title("Peak Fitting with Linear Background")
plt.grid(alpha=0.3)
plt.show()