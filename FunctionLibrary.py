import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from numba import njit
from skimage import exposure
import os
import imageio.v2 as imageio
import pyFAI, fabio
import logging
import json


# --- Utility: Removes the extension from a file name
def remove_filename_extension(path):
    """
    Removes the extension and a trailing '.avg' from a filename.

    Args:
        path (str): The full file path.

    Returns:
        str: The filename without its extension.
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename.replace(".avg","")

# --- Utility: Creates directory for future output data storage
def create_directory(path, logger=None):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The directory path to create.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
    """
    logger = logger or logging.getLogger(__name__)
    try:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created: {path}")
    except Exception as e:
        logger.warning(f"Error creating directory: {e}")
    return path

# --- Utility: Fit pseudo-Voigt to peaks in a .int file ------------------
# .int file is generated as from an ideal alumina (corundum) crystal structure
# from materialsproject.org database (mp-1143) and Vesta to simulate the structure
def fit_peak_centroids(x_data, y_data, height_frac=0.1, distance=20, eta0=0.5, delta_tol=0.1, logger=None):
    """
    Fits pseudo-Voigt profiles to detect peak centroids from 1D XRD data.

    This function first identifies peaks in the intensity data, then fits a
    pseudo-Voigt profile to each peak within a small window to accurately
    determine its centroid position.

    Args:
        x_data (np.ndarray): The dispersive axis data (e.g., q or 2-theta).
        y_data (np.ndarray): The intensity data.
        height_frac (float, optional): Minimum peak height as a fraction of the
            maximum intensity. Defaults to 0.1.
        distance (int, optional): Minimum horizontal distance (in data points)
            between neighboring peaks. Defaults to 20.
        eta0 (float, optional): Initial guess for the eta (mixing) parameter of
            the pseudo-Voigt profile. Defaults to 0.5.
        delta_tol (float, optional): Tolerance for the peak center during fitting,
            defining the bounds relative to the initial guess. Defaults to 0.1.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.

    Returns:
        peak_positions (list): List of fitted centroid positions in the units of x_data.
    """
    logger = logger or logging.getLogger(__name__)

    x = x_data
    y = y_data

    # Find initial peak locations using scipy's find_peaks
    peaks, _ = find_peaks(y, height=np.max(y) * height_frac, distance=distance)
    if len(peaks) == 0:
        logger.warning("No peaks found in the provided data.")
        return [] # Return an empty list if no peaks are found

    dq = x[1] - x[0]  # Calculate the step size of the x-axis
    # Estimate peak widths (FWHM) in terms of x-axis units
    widths_bins = peak_widths(y, peaks, rel_height=0.5)[0]
    widths_x = widths_bins * dq

    peak_positions = []
    for idx, wid in zip(peaks, widths_x):
        x0 = x[idx]  # Initial guess for peak center
        # Define a fitting window around the peak
        half_width = max(2, int(np.ceil(wid / dq)))
        sl = slice(max(0, idx - half_width), min(len(x), idx + half_width + 1))
        try:
            # Set up initial parameters and bounds for the curve fit
            p0 = [y[idx], x0, wid, eta0]
            bounds = ([0, x0 - delta_tol, 0, 0], [np.inf, x0 + delta_tol, np.inf, 1])
            popt, _ = curve_fit(pseudo_voigt, x[sl], y[sl], p0=p0, bounds=bounds)
            peak_positions.append(popt[1])  # Append the fitted centroid (popt[1])
        except Exception:
            logger.exception(f"Fit failed at index {idx} with x0={x0:.2f}")
            continue
    
    # The hardcoded file writing has been removed.
    # The function now only calculates and returns the peak positions.
    return peak_positions

# --- Utility: Convert 2theta from initial fit check into q-space
def convert_2theta_to_q(file_path, wavelength_nm):
    """
    Converts 2θ values from a text file to q values.

    Args:
        file_path (str): Path to the text file containing 2θ values (in degrees).
        wavelength_nm (float): Wavelength in nanometers.

    Returns:
        np.ndarray: Array of q values in nm⁻¹.
    """
    # Load the 2θ values (assuming one per line)
    twotheta_deg = np.loadtxt(file_path)
    
    # Convert 2θ to θ in radians
    theta_rad = np.deg2rad(twotheta_deg / 2.0)
    
    # Calculate q
    q_vals = (4 * np.pi / wavelength_nm) * np.sin(theta_rad)
    
    return q_vals

# --- ImageJ-based autocontrast function ---------------------------------
def imagej_autocontrast(image, k=2.5):
    """
    Adjusts the image contrast based on its statistics, similar to ImageJ's auto-contrast.

    The display range is set to `mean ± k * std_dev`.

    Args:
        image (np.ndarray): The input image.
        k (float, optional): The number of standard deviations to include in the display range. Defaults to 2.5.
    Returns:
        np.ndarray: The contrast-adjusted image, rescaled to the new range.
    """
    mean = np.mean(image)
    std = np.std(image)
    vmin = mean - k * std
    vmax = mean + k * std
    return exposure.rescale_intensity(image, in_range=(vmin, vmax))

# --- Pseudo-Voigt profile -----------------------------------------------
@njit(cache=True)
def pseudo_voigt(x, amp, cen, wid, eta, bg_const): # bg_slope
    """
    A pseudo-Voigt profile, which is a linear combination of Gaussian and Lorentzian profiles.

    The profile is defined as: V(x) = η * L(x) + (1-η) * G(x), where 0 < η < 1.
    The `@njit` decorator from Numba is used for just-in-time compilation to speed up computation.

    Args:
        x (np.ndarray): The independent variable (e.g., q or 2-theta).
        amp (float): The amplitude of the peak.
        cen (float): The center of the peak.
        wid (float): The full width at half maximum (FWHM) of the peak.
        eta (float): The mixing parameter between Gaussian (0) and Lorentzian (1).
        # bg_const (float): Constant (y-intercept) of the background.
        # bg_slope (float): Slope of the background.

    Returns:
        np.ndarray: The calculated pseudo-Voigt profile.
    """
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss   = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    background = bg_const # + bg_slope * (x - cen) # Centering the slope term improves fit stability
    return eta * lorentz + (1 - eta) * gauss + background

# --- PyFAI data loading & integration -----------------------------------
def load_integrator_and_data(poni_path, tif_path, output_path, ref_tif_path=None, mask_threshold=4e2, logger=None, save_adjusted_tif=True):
    """
    Load/initialize PyFAI integrator and adjust a 2D XRD image using an auto-CB 
    scheme from ImageJ. Saves the adjusted image.

    Note: The `ref_tif_path` and `mask_threshold` arguments are currently unused.
    The function returns `None` for the mask.

    Args:
        poni_path (str): Path to the pyFAI calibration file (.poni).
        tif_path (str): Path to the input TIFF image.
        output_path (str): Directory to save the adjusted image.
        ref_tif_path (str, optional): Unused. Defaults to None.
        mask_threshold (float, optional): Unused. Defaults to 4e2.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.

    Returns:
        tuple: A tuple containing (ai, data_adj, mask), where `ai` is the AzimuthalIntegrator,
               `data_adj` is the contrast-adjusted image data, and `mask` is None.
    """
    # Load calibrant and raw image
    ai  = pyFAI.load(poni_path)
    img = fabio.open(tif_path)
    data = img.data.astype(np.float32)

    # Contrast adjustment using ImageJ-style autocontrast (wider dynamic range)
    data_adj_float = imagej_autocontrast(data, k=3.0)

    # Keep data as float32 for pyFAI processing
    data_adj = data_adj_float

    logger = logger or logging.getLogger(__name__)
    if save_adjusted_tif:
        # Save adjusted TIF alongside the original
        base, ext = os.path.splitext(tif_path)
        filename = os.path.basename(base)
        adjusted_path = f"{output_path}/{filename}_adjusted{ext}"
        imageio.imwrite(adjusted_path, data_adj)
        logger.info(f"Adjusted image saved to: {adjusted_path}")
    else:
        logger.info(f"Adjusted image not saved.")

    # This function does not compute a mask; it returns None.
    return ai, data_adj, None

def load_and_prep_image(tif_path, output_path, mask_threshold=4e2, logger=None, save_adjusted_tif=True):
    """
    Loads and prepares a single TIFF image for integration.

    This function is designed for batch processing where the pyFAI integrator
    is already loaded. It performs contrast adjustment and saves the adjusted image.

    Args:
        tif_path (str): Path to the input TIFF image.
        output_path (str): Directory to save the adjusted image.
        mask_threshold (float, optional): Unused. Defaults to 4e2.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.

    Returns:
        tuple: A tuple containing (data_adj, mask), where `data_adj` is the
               contrast-adjusted image data, and `mask` is None.
    """
    logger = logger or logging.getLogger(__name__)

    # Load and process the image data
    img = fabio.open(tif_path)
    data = img.data.astype(np.float32)
    data_adj = imagej_autocontrast(data, k=3.0)

    if save_adjusted_tif:
        # Save the adjusted image
        base, ext = os.path.splitext(tif_path)
        filename = os.path.basename(base)
        adjusted_path = f"{output_path}/{filename}_adjusted{ext}"
        imageio.imwrite(adjusted_path, data_adj)
        logger.info(f"Adjusted image saved to: {adjusted_path}")
    else: 
        logger.info(f"Adjusted image not saved.")


    # This function does not compute a mask; it returns None.
    return data_adj, None

def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000, output_dir=None, save_chi_files=False, logger=None):
    """
    Integrates a 2D diffraction pattern into a 2D (q, chi) representation.

    Optionally saves each azimuthal slice as a .chi file (Fit2D format).

    Args:
        ai (pyFAI.AzimuthalIntegrator): The pyFAI integrator object.
        data (np.ndarray): The 2D diffraction image data.
        mask (np.ndarray): A mask for invalid pixels.
        num_azim_bins (int, optional): Number of azimuthal bins. Defaults to 360.
        q_min (float, optional): Minimum q value for integration. Defaults to 16.0.
        npt_rad (int, optional): Number of points in the radial dimension. Defaults to 5000.
        output_dir (str, optional): Directory to save .chi files. Defaults to None.
        save_chi_files (bool, optional): If True, save .chi files. Defaults to False.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.

    Returns:
        tuple: A tuple containing (I2d, q, chi), where `I2d` is the 2D integrated
               intensity array, `q` is the radial q-values, and `chi` is the
               azimuthal chi-values in degrees.
    """
    if save_chi_files:
        os.makedirs(output_dir, exist_ok=True)

    logger = logger or logging.getLogger(__name__)
    logger.info("Running the integrate_2d() function")
    # Determine q_max from low-res integration
    q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial
    q_max = q_full[-1]

    # Perform the high-resolution 2D integration
    res = ai.integrate2d(
        data,
        npt_rad=npt_rad,
        npt_azim=num_azim_bins,
        unit="q_nm^-1",
        mask=mask,
        radial_range=(q_min, q_max),
    )
    I2d = res.intensity
    q, chi = res.radial, res.azimuthal

    # Ensure I2d has shape (n_azim, n_rad)
    if I2d.shape == (len(q), len(chi)):
        I2d = I2d.T

    # Normalize chi to the range [0, 360) and sort the data accordingly
    chi = (chi + 360) % 360
    order = np.argsort(chi)
    chi = chi[order]
    I2d = I2d[order]

    # Save each azimuthal bin's pattern as a .chi file
    if save_chi_files:
        for i, chi_val in enumerate(chi):
            chi_deg = chi_val
            filename = os.path.join(output_dir, f"azim_{int(round(chi_deg))}deg.chi")
            with open(filename, 'w') as f:
                f.write(f"# Azimuthal bin: {chi_deg:.2f} deg\n")
                f.write("# Columns: q (nm^-1), Intensity (a.u.)\n")
                for q_val, I_val in zip(q, I2d[i]):
                    f.write(f"{q_val:.6f} {I_val:.6f}\n")
        fig_filename = os.path.join(output_dir, "q_vs_chi_plot.png")
        logger.info(f"Stacked q vs chi plot saved to: {fig_filename}")

    return I2d, q, chi

def fit_peaks_with_initial_guesses(I2d, q, q_peaks, delta_tol=0.07, eta0=0.5, n_jobs=-1, delta_array=None, output_dir=None, logger=None):
    """
    Fits each azimuthal slice of I2d using initial q peak guesses and the pseudo-Voigt function.
    This function uses parallel processing to speed up the fitting process.

    Args:
        I2d (np.ndarray): 2D array of intensities [azimuthal_bin x radial_bin].
        q (np.ndarray): 1D array of radial q values.
        q_peaks (list or np.ndarray): Initial guesses for peak positions (q0).
        delta_tol (float, optional): Symmetric tolerance for the fit window around q0. Defaults to 0.07.
        eta0 (float, optional): Initial guess for the pseudo-Voigt mixing parameter. Defaults to 0.5.
        n_jobs (int, optional): Number of parallel jobs for fitting. -1 uses all available cores. Defaults to -1.
        delta_array (np.ndarray, optional): Asymmetric tolerance array of shape (2, N_peaks).
                                           `delta_array[0]` is upper tolerance, `delta_array[1]` is lower.
                                           Overrides `delta_tol`. Defaults to None.
        output_dir (str, optional): Directory to save fitted data. Defaults to None.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.

    Returns:
        tuple: A tuple containing (q_centroids, q_errors, q_chi_path), where `q_centroids`
               is an array of fitted centroid positions, `q_errors` is an array of their
               standard errors, and `q_chi_path` is the path to the saved centroid data file.
    """
    from joblib import Parallel, delayed
    logger = logger or logging.getLogger(__name__)

    dq = q[1] - q[0]
    widths_q = np.full_like(q_peaks, dq * 3)  # reasonable initial width
    # print(delta_array)
    # Define a helper function to fit a single azimuthal slice
    def _fit_slice_w_guesses(intensity_row):
        centroids_out = []
        errors_out = []
        for i, (q0, wid0) in enumerate(zip(q_peaks, widths_q)):
            # Set fit window tolerance (symmetric or asymmetric)
            if delta_array is not None:
                tol_up, tol_dn = delta_array[0][i], delta_array[1][i]
            else:
                tol_up = tol_dn = delta_tol

            mask = (q >= q0 - tol_dn) & (q <= q0 + tol_up)
            x, y = q[mask], intensity_row[mask]
            
            # Skip if there's not enough data to fit
            if len(x) < 5 or not np.any(y > 0):
                centroids_out.append(np.nan)
                errors_out.append(np.nan)
                continue

            try:
                # Perform the curve fit
                bg_const_guess = np.min(y) # Guess the background is at the minimum intensity in the window
                p0 = [np.max(y) - bg_const_guess, q0, wid0, eta0, bg_const_guess]
                # p0 = [np.max(y), q0, wid0, eta0]
                bounds = ([-np.inf, q0 - tol_dn, 0, 0, -np.inf], 
                          [np.inf, q0 + tol_up, np.inf, 1, np.inf])
                
                popt, pcov = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds)
                
                # Extract fitted parameters and their standard errors
                perr = np.sqrt(np.diag(pcov))
                fitted_q = popt[1]
                fitted_q_error = perr[1] # Error of the centroid parameter
                
                edge_buffer = 0
                # Reject fits that are at the edge of the fit window
                if not (q0 - tol_dn + edge_buffer <= fitted_q <= q0 + tol_up - edge_buffer):
                    centroids_out.append(np.nan)
                    errors_out.append(np.nan)
                else:
                    centroids_out.append(fitted_q)
                    errors_out.append(fitted_q_error)

            except (RuntimeError, ValueError): # Catch fitting errors
                logger.debug(f"Fit failed for peak index {i} in azimuthal bin.")
                centroids_out.append(np.nan)
                errors_out.append(np.nan)
        return centroids_out, errors_out

    # Run the fitting in parallel for all azimuthal slices
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice_w_guesses)(row) for row in I2d
    )
    
    q_centroids_list, q_errors_list = zip(*results)

    q_centroids_arr = np.array(q_centroids_list).T
    q_errors_arr = np.array(q_errors_list).T

    # Save the results to text files if an output directory is provided
    if output_dir is not None:
        q_chi_path = os.path.join(output_dir, "q_vs_chi_peaks.txt")
        q_err_path = os.path.join(output_dir, "q_vs_chi_errors.txt")
        
        np.savetxt(q_chi_path, q_centroids_arr, fmt="%.6f", delimiter="\t", header="Rows = rings; Cols = azim bins (q centroids)")
        np.savetxt(q_err_path, q_errors_arr, fmt="%.6f", delimiter="\t", header="Rows = rings; Cols = azim bins (q errors)")
        
        logger.info(f"q vs chi centroid data saved to: {q_chi_path}")
        logger.info(f"q vs chi error data saved to: {q_err_path}")
    else:
        logger.warning("No output directory provided! q vs chi data was not saved!")

    return q_centroids_arr, q_errors_arr, q_chi_path


def plot_q_vs_chi_stacked(file_path, output_dir=None, chi_deg=None, dpi=600, plot=True, calibrant=False, logger=None):
    """
    Plots each row of q_vs_chi_peaks.txt as a stacked subplot, with chi on the x-axis and q on the y-axis.
    This is useful for visualizing the q(χ) variation for each diffraction ring.

    Args:
        file_path (str): Path to the q_vs_chi_peaks.txt file.
        output_dir (str, optional): Directory to save the output plot. Defaults to None.
        chi_deg (ndarray, optional): Azimuthal angle array in degrees. If None, assumes uniform [0, 360).
        dpi (int): Resolution of the saved figure.
        plot (bool, optional): If False, suppresses plotting. Defaults to True.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
    """
    import numpy as np
    logger = logger or logging.getLogger(__name__)

    q_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    num_rings, num_chi = q_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, num_chi, endpoint=False)

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(num_rings, 1, figsize=(8, 2 * num_rings), sharex=True, dpi=dpi)
        for i in range(num_rings):
            ax = axes[i] if num_rings > 1 else axes
            q_vals = q_data[i]
            mask = ~np.isnan(q_vals)
            ax.plot(chi_deg[mask], q_vals[mask], '.', markersize=3)
            ax.set_title(f'Ring {i+1}')
            ax.set_ylabel(f'q (nm⁻¹)')
            ax.set_xlim(0, 360)
            
        axes[-1].set_xlabel('Azimuth χ (°)')
        fig.tight_layout()
        fig_filename = os.path.join(output_dir, "q_vs_chi_plot.png")
        fig.savefig(fig_filename)
        plt.close(fig)
        logger.info(f"Stacked q vs chi plot saved to: {fig_filename}")

# --- Plot strain vs chi stacked (modeled after plot_q_vs_chi_stacked) ---
def plot_strain_vs_chi_stacked(file_path, output_dir=None, chi_deg=None, dpi=600, plot=True, calibrant=False, logger=None):
    """
    Plots each row of strain_vs_chi_peaks.txt as a stacked subplot, with chi on the x-axis and strain on the y-axis.
    This is useful for visualizing the strain(χ) variation for each diffraction ring.

    Args:
        file_path (str): Path to the strain_vs_chi_peaks.txt file.
        output_dir (str, optional): Directory to save the output plot. Defaults to None.
        chi_deg (ndarray, optional): Azimuthal angle array in degrees. If None, assumes uniform [0, 360).
        dpi (int): Resolution of the saved figure.
        plot (bool, optional): If False, suppresses plotting. Defaults to True.
        calibrant (bool, optional): If True, sets a fixed y-axis limit for calibrant data. Defaults to False.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
    """
    logger = logger or logging.getLogger(__name__)
    strain_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    num_rings, num_chi = strain_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, num_chi, endpoint=False)

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(num_rings, 1, figsize=(8, 2 * num_rings), sharex=True, dpi=dpi)
        for i in range(num_rings):
            ax = axes[i] if num_rings > 1 else axes
            vals = strain_data[i]
            mask = ~np.isnan(vals)
            ax.plot(chi_deg[mask], vals[mask], '.', markersize=3)
            ax.set_title(f'Ring {i+1}')
            ax.set_ylabel('Strain')
            ax.set_xlim(0, 360)
            if calibrant:
                ax.set_ylim(-0.0015, 0.0015)
        axes[-1].set_xlabel('Azimuth χ (°)')
        fig.tight_layout()
        scatter_path = os.path.join(output_dir, "strain_vs_chi_plot.png")
        fig.savefig(scatter_path)
        plt.close(fig)
        logger.info(f"Stacked strain vs chi plot saved to: {scatter_path}")

# --- Compute full strain tensor ----------------------------------------
def fit_lattice_cone_distortion(q_data, q_errors, q0_chi_data, initial_q_guesses, wavelength_nm,
                                chi_deg=None, psi_deg=None, phi_deg=None, omega_deg=None, num_strain_components=3, output_dir=None, dpi=600, plot=True, logger=None, min_rsquared=0.0):
    """
    Fits a lattice cone distortion model to q(χ) data to extract strain tensor components.

    This function implements the model described by He & Smith (1998) to determine
    the components of the strain tensor (ε_ij) by fitting the variation of the
    diffraction ring radius (q) with azimuthal angle (χ).

    Args:
        q_data (np.ndarray): Array of q centroids of shape (n_rings, n_chi_bins).
        q_errors (np.ndarray): Array of q centroid errors of shape (n_rings, n_chi_bins).
        q0_chi_data (np.ndarray): Array of unstrained lattice spacings (q0) for each ring.
        initial_q_guesses (list): List of initial q values, used for plot titles.
        wavelength_nm (float): X-ray wavelength in nanometers.
        chi_deg (np.ndarray, optional): Azimuthal angles in degrees. Defaults to a uniform grid.
        psi_deg, phi_deg, omega_deg (np.ndarray, optional): Sample orientation angles. Defaults to a simple geometry.
        num_strain_components (int, optional): Number of strain components to solve for (3, 5, or 6). Defaults to 3.
        output_dir (str, optional): Directory to save plots and results. Defaults to None.
        dpi (int, optional): Resolution for saved plots. Defaults to 600.
        plot (bool, optional): If True, generate and save plots. Defaults to True.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
        min_rsquared (float, optional): Minimum R-squared value to accept a fit. Defaults to 0.5.
    Returns:
        tuple: (strain_params, strain_list, q0_list_out, strain_vs_chi_path)
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    logger = logger or logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    # collect fitted strain-vs-chi curves for overlay
    fit_vs_chi = []

    # q_data is provided as a numpy array of shape (n_rings, n_bins)
    n_rings, n_bins = q_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, n_bins, endpoint=False)
    # set default orientation angles if not provided
    if psi_deg is None:
        psi_deg = np.full_like(chi_deg, 0.0) # ψ = 0°
    if phi_deg is None:
        phi_deg = np.full_like(chi_deg, 0.0) # φ = 0°
    if omega_deg is None:
        omega_deg = np.full_like(chi_deg, 90.0) # ω = 90°

    strain_params = []
    fig, axes = (plt.subplots(n_rings, 1, figsize=(10, 2 * n_rings), dpi=dpi, sharex=True) if plot else (None, None))
    if plot and n_rings == 1: axes = [axes]

    # y limits for the strain vs chi plots
    y_min, y_max = -0.0015, 0.0015

    nan_dict = {key: np.nan for key in ['q0', 'eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz', 
                                        'eps_xx_err', 'eps_xy_err', 'eps_yy_err', 'eps_xz_err', 'eps_yz_err', 'eps_zz_err']}

    for i in range(n_rings):
        q_vals = q_data[i]
        q_errs = q_errors[i]
        q0_vals = q0_chi_data[i]
        
        # Defining q0_fixed for naming of plots
        q0_fixed = initial_q_guesses[i]

        # Filter out any NaN values from the input data
        mask = ~np.isnan(q_vals) & ~np.isnan(q_errs) & ~np.isnan(q0_vals)
        x, y, y_err, q0_masked = chi_deg[mask], q_vals[mask], q_errs[mask], q0_vals[mask]
        
        # Filter outliers using the Median Absolute Deviation (MAD) method
        if len(y) > 0: # Ensure there is data to filter
            median_q = np.median(y)
            abs_deviation = np.abs(y - median_q)
            mad = np.median(abs_deviation)
            
            # Define the outlier threshold (3.0 is a good starting point)
            threshold = 8.0 * mad
            
            # Keep only the points within the threshold
            outlier_mask = abs_deviation < threshold
            
            # Log how many points were removed
            num_outliers = len(y) - np.sum(outlier_mask)
            if num_outliers > 0:
                logger.info(f"Ring {i+1}: Removed {num_outliers} outliers using MAD filter.")
                
            # Apply the mask to your data
            x = x[outlier_mask]
            y = y[outlier_mask]
            y_err = y_err[outlier_mask]
        
        # Ensure a minimum error value to avoid division by zero in weights
        min_error = 1e-6
        y_err = np.maximum(y_err, min_error)

        # Skip ring if there are not enough data points for a reliable fit
        if len(x) < 20:
            logger.warning(f"Ring {i+1}: insufficient data points ({len(x)} < 20). Skipping.")
            if plot:
                axes[i].set_title(f"Ring {i+1}: insufficient data"); axes[i].axis('off')
            strain_params.append(nan_dict.copy()) # creates a dictionary of NaN values if the ring is skipped
            fit_vs_chi.append(np.full(n_bins, np.nan))
            continue

        # q0_fixed = q0_list[i]
        try:
            # --- Geometric Transformations and Model Setup ---
            chi_rad   = np.deg2rad(90-x) # transform so χ is aligned with the coordinate system in He & Smith 1998
            psi_rad   = np.deg2rad(psi_deg[mask])
            phi_rad   = np.deg2rad(phi_deg[mask])
            omega_rad = np.deg2rad(omega_deg[mask])
            # Compute θ for each centroid: θ = arcsin(q*λ/(4π))
            theta     = np.arcsin((y * wavelength_nm) / (4 * np.pi))
            sin_chi   = np.sin(chi_rad)
            cos_chi   = np.cos(chi_rad)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            # Intermediate parameters a, b, c from Table 1 He & Smith 1998
            a = sin_theta * np.cos(omega_rad) + sin_chi * cos_theta * np.sin(omega_rad)
            b = -cos_chi * cos_theta
            c = sin_theta * np.sin(omega_rad) - sin_chi * cos_theta * np.cos(omega_rad)
            # Direction cosines A, B, C
            A = a * np.cos(phi_rad) - b * np.cos(psi_rad) * np.sin(phi_rad) + c * np.sin(psi_rad) * np.sin(phi_rad)
            B = a * np.sin(phi_rad) + b * np.cos(psi_rad) * np.cos(phi_rad) - c * np.sin(psi_rad) * np.cos(phi_rad)
            C = b * np.sin(psi_rad) + c * np.cos(psi_rad)
            # Defining the strain coefficients: f_ij
            f11, f12, f22, f13, f23, f33 = A**2, 2*A*B, B**2, 2*A*C, 2*B*C, C**2
            
            # The model is y_meas = F * [strain_components]
            F_all = np.vstack([f11, f12, f22, f13, f23, f33]).T
            y_meas = np.log(q0_masked / y)
            
            # Propagate errors to get weights for the Weighted Least Squares (WLS) fit
            # Error in y_meas = ln(q0/q) is approx. sigma_q / q
            y_meas_err = y_err / y
            y_meas_err[y_meas_err == 0] = min_error # Avoid division by zero
            weights = 1.0 / (y_meas_err**2)
            
            epsilon = '\u03B5' # ε
            logger.info(f"The model is solving for {num_strain_components} strain components.")
            # Select the appropriate columns of the design matrix F based on the desired model
            if num_strain_components == 6:
                F = F_all
                param_names = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']
                if i==1: logger.info(f"No strain components are set to 0")
            elif num_strain_components == 5:
                F = F_all[:, [0, 1, 2, 3, 4]]
                param_names = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz']
                if i==1: logger.info(f"{epsilon}33 is set to zero")
            elif num_strain_components == 3:
                F = F_all[:, [0, 1, 2]]
                param_names = ['eps_xx', 'eps_xy', 'eps_yy']
                if i==1: logger.info(f"{epsilon}13, {epsilon}23, & {epsilon}33 are set to zero")
            else: 
                F = F_all[:, [0, 1, 2]]
                param_names = ['eps_xx', 'eps_xy', 'eps_yy']
                if i==1: logger.warning(f"An incompatible number of strain components was selected. The only valid numbers are 3 (biaxial), 5 (biaxial w/ shear) & 6 (full strain tensor). Defaulting to biaxial.")

            # --- Perform WLS fit and process results ---
            wls_model = sm.WLS(y_meas, F, weights=weights)
            results = wls_model.fit()
            
            # Filter out poor fits based on the R-squared value
            if results.rsquared < min_rsquared:
                raise ValueError(f"R-squared ({results.rsquared:.3f}) is below the threshold of {min_rsquared}.")

            # Store results and errors
            params = {f'{p}': val for p, val in zip(param_names, results.params)}
            errors = {f'{p}_err': err for p, err in zip(param_names, results.bse)}
            
            # Fill in NaNs for components not in the fit
            full_params = nan_dict.copy()
            full_params['q0'] = np.mean(q0_masked)
            full_params.update(params)
            full_params.update(errors)
            
            strain_params.append(full_params)

            if plot:
                ax = axes[i]
                ax.plot(x, y_meas, '.', markersize=3, label='ln(q₀/q)')
                ax.plot(x, results.fittedvalues, '-', label='Fit') # Use results.fittedvalues
                ax.set_ylabel(f'{epsilon}=ln(q₀/q)')
                ax.set_title(f'Ring {i+1} (q₀ = {q0_fixed:.4f} nm⁻¹)')
                ax.legend(fontsize='small', loc='lower left', bbox_to_anchor=(1.02, 0.02))
        except (ValueError, np.linalg.LinAlgError):
            if plot:
                axes[i].set_title(f"Ring {i+1} (q₀ = {q0_fixed:.4f} nm⁻¹): fit failed"); axes[i].axis('off')
            strain_params.append(nan_dict.copy())
            fit_vs_chi.append(np.full(n_bins, np.nan))
            logger.exception(f"Fit failed for Ring {i+1}")

    if plot:
        axes[-1].set_xlabel('Azimuth χ (°)')
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "strain_vs_chi_plot_fitted.png")
        fig.savefig(fig_path)
        plt.close('all')
        logger.info(f"Combined distortion fit plot saved to: {fig_path}")

    # Calculate the average strain for each ring
    strain_list = []
    for i, row in enumerate(q_data):
        mask = ~np.isnan(row)
        q_avg = np.mean(row[mask]) if np.any(mask) else np.nan
        q0 = strain_params[i].get('q0', np.nan)
        if not np.isnan(q0) and q0 != 0 and not np.isnan(q_avg):
            strain = (q0 - q_avg) / q0
        else:
            strain = np.nan
        strain_list.append(strain)

    # Extract the list of q0 values used for the fits
    q0_list_out = [p.get('q0', np.nan) for p in strain_params]

    # Save the full strain vs chi array
    strain_vs_chi = (q0_chi_data - q_data) / q0_chi_data
    strain_vs_chi_path = os.path.join(output_dir, "strain_vs_chi_peaks.txt")
    np.savetxt(strain_vs_chi_path, strain_vs_chi, fmt="%.6e", delimiter="\t",
               header="Rows = diffraction rings; Columns = azimuthal bins (strain vs chi data)")
    logger.info(f"Strain vs chi centroid data saved to: {strain_vs_chi_path}")

    return strain_params, strain_list, q0_list_out, strain_vs_chi_path

def calculate_and_log_map_error_metrics(data_map, error_map, map_name, logger, file_handle=None):
    """
    Calculates error metrics for a given data map and its corresponding error map.

    Metrics include average error, RMS error, and the standard deviation of the data itself.

    Args:
        data_map (np.ndarray): The 2D map of data values.
        error_map (np.ndarray): The 2D map of measurement errors.
        map_name (str): The name of the map for logging purposes.
        logger (logging.Logger): Logger for console output.
        file_handle (file object, optional): A file handle to write the metrics to. Defaults to None.
    """
    import numpy as np

    valid_errors = error_map[~np.isnan(error_map)]
    valid_data = data_map[~np.isnan(data_map)]

    if valid_errors.size == 0:
        logger.warning(f"No valid data to calculate error metrics for {map_name}.")
        return

    avg_error = np.mean(valid_errors)
    rms_error = np.sqrt(np.mean(valid_errors**2))
    std_dev_of_values = np.std(valid_data)

    # --- Create the output string ---
    header = f"--- Error Metrics for: {map_name} ---"
    metrics_str = (
        f"{header}\n"
        f"    Average Measurement Error: {avg_error:.3e}\n"
        f"    RMS of Measurement Error:  {rms_error:.3e}\n"
        f"    Standard Deviation of Map Values (Spatial Variation): {std_dev_of_values:.3e}\n"
        f"{'-' * len(header)}\n"
    )

    # Log to console/main log file
    logger.info(metrics_str)

    # --- Write to the summary text file if a handle is provided ---
    if file_handle:
        file_handle.write(metrics_str + "\n")

# --- Generate strain maps from JSON ---------------------------------
# Generates the strain maps without the need to have a different function for contiguous and
# non-contiguous scan layouts
def generate_strain_maps_from_json(
    json_path,
    n_rows,
    n_cols,
    step_size,
    pixel_size_map,
    start_xy=(0.0, 0.0),
    gap_mm=None,
    color_limit_window=None,
    map_offset_xy=(0.0, 0.0),
    trim_edges=False,
    title_and_labels=True,
    colorbar_scale=None,
    output_dir="StrainMaps",
    dpi=600,
    map_name_pfx="strain-map_",
    logger=None,
    num_strain_components=3
):
    """
    Generates and saves strain maps from a JSON file containing strain tensor data.

    This function reads strain tensor components for a grid of measurement points,
    reshapes them into 2D maps, and plots them as physically scaled images. It also
    calculates and saves a summary of error metrics.

    Args:
        json_path (str): Path to the input JSON file.
        n_rows (int): Number of rows in the measurement grid.
        n_cols (int): Number of columns in the measurement grid.
        step_size (tuple): (dX, dY) step size in millimeters.
        pixel_size_map (tuple): (width, height) of each measurement point's representation on the map.
        start_xy (tuple, optional): (X, Y) coordinates of the first point (top-left). Defaults to (0.0, 0.0).
        gap_mm (float, optional): Additional gap between columns. Defaults to None.
        color_limit_window (tuple, optional): (x_min, x_max) in mm to define a window for color scale limits and error calculation. Defaults to None.
        map_offset_xy (tuple, optional): (X, Y) offset to apply to the entire map. Defaults to (0.0, 0.0).
        trim_edges (bool, optional): If True, trims map edges to not go below zero. Defaults to False.
        title_and_labels (bool, optional): If False, hides titles and labels on plots. Defaults to True.
        colorbar_scale (tuple, optional): (vmin, vmax) for the color bar. Defaults to auto-scaling.
        output_dir (str, optional): Directory to save maps and summary. Defaults to "StrainMaps".
        num_strain_components (int, optional): Number of strain components to plot (3, 5, or 6). Defaults to 3.
    """
    import os, numpy as np, json, time
    import matplotlib.pyplot as plt, matplotlib.patches as patches, matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker as mticker
    from joblib import Parallel, delayed
    
    logger = logger or logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        strain_data = json.load(f)

    # Data validation
    num_points = len(strain_data)
    if num_points != n_rows * n_cols:
        raise ValueError(f"Grid dimension mismatch! JSON data points ({num_points}) do not match grid ({n_rows}x{n_cols}).")
    logger.info(f"Loaded {num_points} data points for a {n_rows}x{n_cols} grid.")

    num_rings = len(strain_data[0].get("strain_tensor", []))
    if gap_mm is None: gap_mm = 0.0

    # --- Data Extraction ---
    value_keys = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']
    error_keys = [f'{key}_err' for key in value_keys]

    filtered_values = [[] for _ in range(num_rings)]
    filtered_errors = [[] for _ in range(num_rings)]

    for entry in strain_data:
        tensors = entry.get("strain_tensor", [])
        for i in range(num_rings):
            if i < len(tensors) and isinstance(tensors[i], dict):
                filtered_values[i].append([tensors[i].get(k, np.nan) for k in value_keys])
                filtered_errors[i].append([tensors[i].get(k, np.nan) for k in error_keys])
            else:
                filtered_values[i].append([np.nan] * 6)
                filtered_errors[i].append([np.nan] * 6)

    # Select components to plot based on the input parameter
    if num_strain_components == 6:
        components_to_plot = ['xx', 'xy', 'yy', 'xz', 'yz', 'zz']
    elif num_strain_components == 5:
        components_to_plot = ['xx', 'xy', 'yy', 'xz', 'yz']
    else:  # Default to 3 components (biaxial)
        components_to_plot = ['xx', 'xy', 'yy']
        if num_strain_components != 3:
            logger.warning(f"Invalid num_strain_components ({num_strain_components}). Defaulting to 3.")
    
    all_components = [
        {'name': 'xx', 'index': 0, 'title': 'Strain_xx', 'latex': r'$\varepsilon_{xx}$'},
        {'name': 'xy', 'index': 1, 'title': 'Strain_xy', 'latex': r'$\varepsilon_{xy}$'},
        {'name': 'yy', 'index': 2, 'title': 'Strain_yy', 'latex': r'$\varepsilon_{yy}$'},
        {'name': 'xz', 'index': 3, 'title': 'Strain_xz', 'latex': r'$\varepsilon_{xz}$'},
        {'name': 'yz', 'index': 4, 'title': 'Strain_yz', 'latex': r'$\varepsilon_{yz}$'},
        {'name': 'zz', 'index': 5, 'title': 'Strain_zz', 'latex': r'$\varepsilon_{zz}$'}
    ]
    components_to_process = [c for c in all_components if c['name'] in components_to_plot]

    # --- Map Geometry Setup ---
    pixel_size_unit = "mm"
    startX, startY = start_xy
    shiftX, shiftY = map_offset_xy
    startX += shiftX
    startY += shiftY
    
    dX, dY = step_size
    
    # Pre-calculate column indices for the color limit window if specified
    win_idx0, win_idx1 = None, None
    if color_limit_window:
        x_min_win, x_max_win = color_limit_window
        # Adjust window by the same offset to sample the correct region
        x_min_win_shifted, x_max_win_shifted = x_min_win + shiftX, x_max_win + shiftX
        
        col_step = dX + gap_mm
        win_idx0 = max(0, int(np.floor((x_min_win_shifted - startX) / col_step)))
        win_idx1 = min(n_cols, int(np.ceil((x_max_win_shifted - startX) / col_step)))
        logger.info(f"Error metrics will be calculated based on the window defined by columns {win_idx0} to {win_idx1}.")

    # --- Plotting Function ---
    def plot_and_save(data, title, filename):
        data = np.flipud(data)

        fig, ax = plt.subplots(figsize=(3.5, 4), dpi=dpi)
        cmap = plt.cm.jet.copy()
        
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        # Determine color scale limits, either from a window or the full map
        if color_limit_window:
            x_min_win, x_max_win = color_limit_window
            x_min_win_shifted, x_max_win_shifted = x_min_win + shiftX, x_max_win + shiftX
            
            idx0 = max(0, int(np.floor((x_min_win_shifted - startX) / (dX + gap_mm))))
            idx1 = min(n_cols, int(np.ceil((x_max_win_shifted - startX) / (dX + gap_mm))))
            subset = data[:, idx0:idx1]
            if np.any(~np.isnan(subset)):
                data_min, data_max = np.nanmin(subset), np.nanmax(subset)
        
        if colorbar_scale:
            norm = mcolors.Normalize(vmin=colorbar_scale[0], vmax=colorbar_scale[1])
        else:
            norm = mcolors.Normalize(vmin=data_min, vmax=data_max)

        pixel_width, pixel_height = pixel_size_map

        # Draw a colored rectangle for each data point
        for i in range(n_rows):
            for j in range(n_cols):
                strain_val = data[i, j]
                if np.isnan(strain_val): continue
                
                center_x = startX + j * (dX + gap_mm)
                center_y = startY + i * dY
                
                bottom_left_x = center_x - (pixel_width / 2)
                bottom_left_y = center_y - (pixel_height / 2)
                
                rect = patches.Rectangle((bottom_left_x, bottom_left_y), pixel_width, pixel_height, facecolor=cmap(norm(strain_val)))
                ax.add_patch(rect)

        # Set map boundaries
        x_min_edge = startX - (pixel_width / 2)
        x_max_edge = startX + (n_cols - 1) * (dX + gap_mm) + (pixel_width / 2)        
        y_max_edge = startY - (pixel_height / 2)
        y_min_edge = startY + (n_rows - 1) * dY + (pixel_height / 2)
        
        if trim_edges:
            x_min_edge = max(x_min_edge, 0.0)
            y_min_edge = max(y_min_edge, 0.0)
            
        ax.set_xlim(x_min_edge, x_max_edge)
        ax.set_ylim(y_max_edge, y_min_edge)
        ax.set_aspect('equal', adjustable='box')
        
        # Add colorbar and labels
        sm = ScalarMappable(cmap=cmap, norm=norm)
        if title_and_labels:
            cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
            cb.set_label('Microstrain [με]')
            cb.formatter = FuncFormatter(lambda x, _: f"{(x * 1e6):.0f}")
            cb.locator = mticker.MaxNLocator(nbins=9)
            cb.update_ticks()

            ax.set_title(title)
            ax.set_xlabel(f'X Position [{pixel_size_unit}]')
            ax.set_ylabel(f'Y Position [{pixel_size_unit}]')
        plt.tight_layout()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"{title} heatmap saved to: {filepath}")

    # --- Main Processing Logic ---
    def _plot_one_ring(ring_index, value_data, error_data):
        strain_array = np.array(value_data).reshape((n_rows, n_cols, 6))
        
        for comp in components_to_process:
            name, index, title_latex = comp['name'], comp['index'], comp['latex']
            data_map = strain_array[:, :, index]
            title = f"{title_latex} (Ring {ring_index + 1})"
            filename = f"{map_name_pfx}_{name}_ring{ring_index + 1}.png"
            plot_and_save(data_map, title, filename)
        
        eps_xx = strain_array[:,:,0] if 'xx' in components_to_plot else 0.0
        eps_xy = strain_array[:,:,1] if 'xy' in components_to_plot else 0.0
        eps_yy = strain_array[:,:,2] if 'yy' in components_to_plot else 0.0
        eps_xz = strain_array[:,:,3] if 'xz' in components_to_plot else 0.0
        eps_yz = strain_array[:,:,4] if 'yz' in components_to_plot else 0.0
        eps_zz = strain_array[:,:,5] if 'zz' in components_to_plot else 0.0
        
        # Calculate von Mises equivalent strain
        vm_strain = np.sqrt(((eps_xx - eps_yy)**2 + (eps_yy - eps_zz)**2 + (eps_zz - eps_xx)**2) / 2 + 3 * (eps_xy**2 + eps_xz**2 + eps_yz**2))
        title_vm = f"$\\varepsilon_{{VM}}$ (Ring {ring_index + 1})"
        filename_vm = f"{map_name_pfx}_Mises_ring{ring_index + 1}.png"
        plot_and_save(vm_strain, title_vm, filename_vm)

    # Plot maps for each ring in parallel
    Parallel(n_jobs=-1)(delayed(_plot_one_ring)(i, filtered_values[i], filtered_errors[i]) for i in range(num_rings))

    # --- Error Metrics Summary ---
    summary_filepath = os.path.join(output_dir, "error_metrics_summary.txt")
    with open(summary_filepath, 'w') as f:
        f.write(f"Error Metrics Summary for: {map_name_pfx}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if color_limit_window:
            f.write(f"Metrics calculated for data within x-window: {color_limit_window} mm\n")
        f.write("="*50 + "\n\n")

        f.write("--- Metrics for Each Ring ---\n\n")
        for i in range(num_rings):
            strain_array = np.array(filtered_values[i]).reshape((n_rows, n_cols, 6))
            error_array = np.array(filtered_errors[i]).reshape((n_rows, n_cols, 6))

            for comp in components_to_process:
                data_map = strain_array[:, :, comp['index']]
                error_map = error_array[:, :, comp['index']]
                map_name = f"{comp['title']} Ring {i+1}"
                
                # Use the windowed subset for calculation if defined
                if win_idx0 is not None:
                    data_subset = data_map[:, win_idx0:win_idx1]
                    error_subset = error_map[:, win_idx0:win_idx1]
                    map_name_for_calc = f"{map_name} (Windowed)"
                else:
                    data_subset = data_map
                    error_subset = error_map
                    map_name_for_calc = map_name
                
                calculate_and_log_map_error_metrics(data_subset, error_subset, map_name_for_calc, logger, file_handle=f)
            
            eps_xx, eps_yy, eps_zz, eps_xy = strain_array[:,:,0], strain_array[:,:,2], strain_array[:,:,5], strain_array[:,:,1]
            err_xx, err_yy, err_zz, err_xy = error_array[:,:,0], error_array[:,:,2], error_array[:,:,5], error_array[:,:,1]
            vm_strain = np.sqrt(((eps_xx - eps_yy)**2 + (eps_yy - eps_zz)**2 + (eps_zz - eps_xx)**2) / 2 + 3 * (eps_xy**2))
            # Note: This error propagation for von Mises is a simple approximation
            err_vm = np.sqrt(err_xx**2 + err_yy**2 + err_zz**2 + err_xy**2)
            
            # Apply slicing for von Mises calculation as well
            if win_idx0 is not None:
                vm_strain_subset = vm_strain[:, win_idx0:win_idx1]
                err_vm_subset = err_vm[:, win_idx0:win_idx1]
                map_name_for_calc = f"Strain_VM Ring {i+1} (Windowed)"
            else:
                vm_strain_subset = vm_strain
                err_vm_subset = err_vm
                map_name_for_calc = f"Strain_VM Ring {i+1}"

            calculate_and_log_map_error_metrics(vm_strain_subset, err_vm_subset, map_name_for_calc, logger, file_handle=f)

        # --- Averaged Maps and Metrics ---
        f.write("\n--- Metrics for Averaged Maps ---\n\n")
        avg_maps = {}
        avg_error_maps = {}
        for comp in components_to_process:
            name, index, title = comp['name'], comp['index'], comp['title']
            avg_map = np.nanmean([np.array(r)[:, index].reshape(n_rows, n_cols) for r in filtered_values], axis=0)
            avg_maps[name] = avg_map
            error_maps = [np.array(r)[:, index].reshape(n_rows, n_cols) for r in filtered_errors]
            # Combine errors in quadrature
            avg_error_map = np.sqrt(np.nanmean(np.square(error_maps), axis=0))
            avg_error_maps[name] = avg_error_map

            if win_idx0 is not None:
                avg_map_subset = avg_map[:, win_idx0:win_idx1]
                avg_error_map_subset = avg_error_map[:, win_idx0:win_idx1]
                map_name_for_calc = f"{title} (Windowed)"
            else:
                avg_map_subset = avg_map
                avg_error_map_subset = avg_error_map
                map_name_for_calc = title
            calculate_and_log_map_error_metrics(avg_map_subset, avg_error_map_subset, map_name_for_calc, logger, file_handle=f)

        # Plot the final averaged maps
        for comp in components_to_process:
            name = comp['name']
            latex_title = comp['latex']
            filename = f"{map_name_pfx}_{name}_avg.png"
            plot_and_save(avg_maps[name], f"{latex_title} (Avg)", filename)
        
        vm_title = 'Strain_VM (Avg)'
        vm_title_latex = r'$\varepsilon_{VM}$ (Avg)'
        
        s_xx = avg_maps.get('xx', np.nan)
        s_xy = avg_maps.get('xy', np.nan)
        s_yy = avg_maps.get('yy', np.nan)
        s_xz = avg_maps.get('xz', 0.0)
        s_yz = avg_maps.get('yz', 0.0)
        s_zz = avg_maps.get('zz', 0.0)

        # Calculate average von Mises strain
        avg_vm_strain = np.sqrt(0.5 * ((s_xx-s_yy)**2 + (s_yy-s_zz)**2 + (s_zz-s_xx)**2) + 3*(s_xy**2 + s_xz**2 + s_yz**2))

        err_xx = avg_error_maps.get('xx', np.nan)
        err_xy = avg_error_maps.get('xy', np.nan)
        err_yy = avg_error_maps.get('yy', np.nan)
        err_xz = avg_error_maps.get('xz', 0.0)
        err_yz = avg_error_maps.get('yz', 0.0)
        err_zz = avg_error_maps.get('zz', 0.0)
        
        # Approximate error for average von Mises strain
        avg_vm_error = np.sqrt(err_xx**2 + err_yy**2 + err_zz**2 + err_xy**2 + err_xz**2 + err_yz**2)

        if win_idx0 is not None:
            avg_vm_strain_subset = avg_vm_strain[:, win_idx0:win_idx1]
            avg_vm_error_subset = avg_vm_error[:, win_idx0:win_idx1]
            map_name_for_calc = f"{vm_title} (Windowed)"
        else:
            avg_vm_strain_subset = avg_vm_strain
            avg_vm_error_subset = avg_vm_error
            map_name_for_calc = vm_title

        calculate_and_log_map_error_metrics(avg_vm_strain_subset, avg_vm_error_subset, map_name_for_calc, logger, file_handle=f)

        plot_and_save(avg_vm_strain, vm_title_latex, f"{map_name_pfx}_Mises_avg.png")

    logger.info("Averaged maps plotted and error summary file saved.")

# --- Utility: Reconstruct simulated diffraction rings using fitted strain tensor components ---
def reconstruct_rings_from_json(json_path, wavelength_nm, chi_step=1.0, logger=None, plot=True, output_dir=None):
    """
    Reconstruct simulated diffraction rings using fitted strain tensor components.
    This function takes the fitted strain tensor from a JSON file and calculates the
    expected q(χ) variation for each diffraction ring.

    Args:
        json_path (str): Path to the JSON file with strain tensor data.
        wavelength_nm (float): X-ray wavelength in nanometers.
        chi_step (float, optional): Step size in degrees for the χ grid. Defaults to 1.0.
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
        plot (bool, optional): If True, generate and save plots. Defaults to True.
        output_dir (str, optional): Directory to save plots. Defaults to None.

    Returns:
        dict: A dictionary where keys are ring indices and values are lists of (chi_deg, q_sim) tuples.
    """
    import matplotlib.pyplot as plt

    with open(json_path, "r") as f:
        strain_data = json.load(f)

    chi_deg = np.arange(0, 360, chi_step)
    # Apply the same convention correction as in fit_lattice_cone_distortion
    chi_rad = np.deg2rad(90.0 - chi_deg)
    results = {}

    for entry in strain_data:
        if "strain_tensor" not in entry:
            continue
        # --- Reconstruct q(chi) for each ring in the entry ---
        tensors = entry["strain_tensor"]
        for i, tensor in enumerate(tensors):
            eps_xx = tensor.get("eps_xx", np.nan)
            eps_xy = tensor.get("eps_xy", np.nan)
            eps_yy = tensor.get("eps_yy", np.nan)
            eps_xz = tensor.get("eps_xz", np.nan)
            eps_yz = tensor.get("eps_yz", np.nan)
            eps_zz = tensor.get("eps_zz", np.nan)
            q0 = tensor.get("q0", None)
            if q0 is None:
                if logger:
                    logger.warning(f"No q0 value found for ring {i+1}; skipping reconstruction.")
                continue

            # --- Apply the forward model from fit_lattice_cone_distortion ---
            phi_rad = 0.0
            psi_rad = 0.0
            omega_rad = np.deg2rad(90.0)

            theta = np.arcsin((q0 * wavelength_nm) / (4 * np.pi))
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)

            a = sin_theta * np.cos(omega_rad) + np.sin(chi_rad) * cos_theta * np.sin(omega_rad)
            b = -np.cos(chi_rad) * cos_theta
            c = sin_theta * np.sin(omega_rad) - np.sin(chi_rad) * cos_theta * np.cos(omega_rad)

            A = a*np.cos(phi_rad) - b*np.cos(psi_rad)*np.sin(phi_rad) + c*np.sin(psi_rad)*np.sin(phi_rad)
            B = a*np.sin(phi_rad) + b*np.cos(psi_rad)*np.cos(phi_rad) - c*np.sin(psi_rad)*np.cos(phi_rad)
            C = b*np.sin(psi_rad) + c*np.cos(psi_rad)

            f11 = A**2
            f22 = B**2
            f33 = C**2
            f12 = 2 * A * B
            f13 = 2 * A * C
            f23 = 2 * B * C

            strain_terms = (f11 * eps_xx + f22 * eps_yy + f33 * eps_zz +
                            f12 * eps_xy + f13 * eps_xz + f23 * eps_yz)

            # Calculate the simulated q-value based on the strain
            q_sim = q0 * np.exp(-strain_terms)
            if i+1 not in results:
                results[i+1] = []
            results[i+1].append((chi_deg, q_sim.tolist()))

            if plot:
                plt.figure()
                plt.plot(chi_deg, q_sim, label=f"Ring {i+1} Simulated")
                plt.xlabel("χ (deg)")
                plt.ylabel("q (1/nm)")
                plt.title(f"Reconstructed Ring {i+1}")
                plt.legend()
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, f"ring_{i+1}_simulated.png"), dpi=600)
                plt.close()

    return results

# --- Generate stress maps from JSON ---
def generate_stress_maps_from_json(
    json_path,
    youngs_modulus,
    poissons_ratio,
    n_rows,
    n_cols,
    step_size,
    pixel_size_map,
    start_xy=(0.0, 0.0),
    gap_mm=None,
    color_limit_window=None,
    map_offset_xy=(0.0, 0.0),
    trim_edges=False,
    colorbar_scale=None,
    output_dir="StressMaps",
    dpi=600,
    map_name_pfx="stress-map_",
    logger=None,
):
    """
    Calculates stress from strain data in a JSON file and generates physically accurate stress maps.

    Assumes an isotropic, linear elastic material under a plane strain condition (eps_zz = 0).

    Args:
        json_path (str): Path to the input JSON file with strain data.
        youngs_modulus (float): Young's Modulus in GPa.
        poissons_ratio (float): Poisson's ratio.
        n_rows (int): Number of rows in the measurement grid.
        n_cols (int): Number of columns in the measurement grid.
        step_size (tuple): (dX, dY) step size in millimeters.
        pixel_size_map (tuple): (width, height) of each measurement point's representation.
        start_xy (tuple, optional): (X, Y) coordinates of the first point. Defaults to (0.0, 0.0).
        gap_mm (float, optional): Additional gap between columns. Defaults to None.
        color_limit_window (tuple, optional): (x_min, x_max) for color scale limits. Defaults to None.
        map_offset_xy (tuple, optional): (X, Y) offset for the map. Defaults to (0.0, 0.0).
        output_dir (str, optional): Directory to save maps. Defaults to "StressMaps".
        logger (logging.Logger, optional): Logger for status messages. Defaults to None.
    """
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter
    from matplotlib.cm import ScalarMappable
    from joblib import Parallel, delayed

    logger = logger or logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        strain_data = json.load(f)
    
    # --- Strain to Stress Conversion using Hooke's Law for Isotropic Materials ---
    E = youngs_modulus * 1e9  # Convert GPa to Pa
    nu = poissons_ratio

    # Lamé parameters
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))  # Shear modulus

    stress_data = []
    for entry in strain_data:
        stress_tensors = []
        for strain_tensor in entry['strain_tensor']:
            eps_xx = strain_tensor.get('eps_xx', np.nan)
            eps_xy = strain_tensor.get('eps_xy', np.nan)
            eps_yy = strain_tensor.get('eps_yy', np.nan)
            eps_xz = strain_tensor.get('eps_xz', np.nan)
            eps_yz = strain_tensor.get('eps_yz', np.nan)
            eps_zz = 0.0 # Assuming plane strain as per the data format
            
            if np.isnan(eps_xx): # If strain is NaN, stress is also NaN
                stress_tensors.append({k: np.nan for k in ['sigma_xx', 'sigma_xy', 'sigma_yy', 'sigma_xz', 'sigma_yz', 'sigma_zz']})
                continue
            
            trace_eps = eps_xx + eps_yy + eps_zz
            
            # Hooke's Law for isotropic material (in Pa)
            sigma_xx = 2 * G * eps_xx + lam * trace_eps
            sigma_yy = 2 * G * eps_yy + lam * trace_eps
            sigma_zz = lam * trace_eps # Non-zero for plane strain
            sigma_xy = 2 * G * eps_xy
            sigma_xz = 2 * G * eps_xz
            sigma_yz = 2 * G * eps_yz

            stress_tensors.append({
                "sigma_xx": sigma_xx / 1e6, "sigma_xy": sigma_xy / 1e6, # Convert to MPa
                "sigma_yy": sigma_yy / 1e6, "sigma_xz": sigma_xz / 1e6,
                "sigma_yz": sigma_yz / 1e6, "sigma_zz": sigma_zz / 1e6,
            })
        stress_data.append(stress_tensors)

    # --- Map Generation (similar logic to generate_strain_maps_from_json) ---
    num_points = len(stress_data)
    if num_points != n_rows * n_cols:
        raise ValueError(f"Grid dimension mismatch! JSON data points ({num_points}) do not match grid ({n_rows}x{n_cols}).")
    logger.info(f"Calculated stress for {num_points} points on a {n_rows}x{n_cols} grid.")

    num_rings = len(stress_data[0]) if stress_data else 0
    if gap_mm is None: gap_mm = 0.0

    # Reshape the flat list of stress tensors into a list of lists for each ring
    filtered = [[] for _ in range(num_rings)]
    for tensors in stress_data:
        for i in range(num_rings):
            if i < len(tensors) and isinstance(tensors[i], dict):
                filtered[i].append([
                    tensors[i].get("sigma_xx", np.nan), tensors[i].get("sigma_xy", np.nan),
                    tensors[i].get("sigma_yy", np.nan), tensors[i].get("sigma_xz", np.nan),
                    tensors[i].get("sigma_yz", np.nan), tensors[i].get("sigma_zz", np.nan)
                ])
            else:
                filtered[i].append([np.nan] * 6)

    pixel_size_unit = "mm"
    startX, startY = start_xy
    shiftX, shiftY = map_offset_xy
    startX += shiftX
    startY += shiftY
    dX, dY = step_size

    # --- Plotting Function ---
    def plot_and_save(data, title, filename):
        data = np.flipud(data)
        fig, ax = plt.subplots(figsize=(3.5, 4), dpi=dpi)
        cmap = plt.cm.jet.copy()
        
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if color_limit_window:
            x_min_win, x_max_win = color_limit_window
            x_min_win_shifted, x_max_win_shifted = x_min_win + shiftX, x_max_win + shiftX
            idx0 = max(0, int(np.floor((x_min_win_shifted - startX) / (dX + gap_mm))))
            idx1 = min(n_cols, int(np.ceil((x_max_win_shifted - startX) / (dX + gap_mm))))
            subset = data[:, idx0:idx1]
            if np.any(~np.isnan(subset)):
                data_min, data_max = np.nanmin(subset), np.nanmax(subset)
        
        norm = mcolors.Normalize(vmin=colorbar_scale[0] if colorbar_scale else data_min,
                                 vmax=colorbar_scale[1] if colorbar_scale else data_max)

        pixel_width, pixel_height = pixel_size_map

        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i, j]
                if np.isnan(val): continue
                center_x, center_y = startX + j * (dX + gap_mm), startY + i * dY
                bottom_left_x, bottom_left_y = center_x - (pixel_width / 2), center_y - (pixel_height / 2)
                rect = patches.Rectangle((bottom_left_x, bottom_left_y), pixel_width, pixel_height, facecolor=cmap(norm(val)))
                ax.add_patch(rect)

        x_min_edge = startX - (pixel_width / 2)
        x_max_edge = startX + (n_cols - 1) * (dX + gap_mm) + (pixel_width / 2)        
        y_max_edge = startY - (pixel_height / 2)
        y_min_edge = startY + (n_rows - 1) * dY + (pixel_height / 2)
        
        if trim_edges:
            x_min_edge = max(x_min_edge, 0.0)
            y_min_edge = max(y_min_edge, 0.0)
            
        ax.set_xlim(x_min_edge, x_max_edge)
        ax.set_ylim(y_max_edge, y_min_edge)
        ax.set_aspect('equal', adjustable='box')
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cb.set_label('Stress (MPa)')
        cb.update_ticks()

        ax.set_title(title)
        ax.set_xlabel(f'X Position [{pixel_size_unit}]')
        ax.set_ylabel(f'Y Position [{pixel_size_unit}]')
        plt.tight_layout()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"{title} heatmap saved to: {filepath}")

    # --- Main Processing Logic ---
    def _plot_one_ring(ring_index, ring_data):
        stress_array = np.array(ring_data).reshape((n_rows, n_cols, 6))
        
        sigma_xx = stress_array[:, :, 0]
        sigma_xy = stress_array[:, :, 1]
        sigma_yy = stress_array[:, :, 2]
        sigma_xz = stress_array[:, :, 3]
        sigma_yz = stress_array[:, :, 4]
        sigma_zz = stress_array[:, :, 5]
        
        # Von Mises Stress Calculation
        vm_stress = np.sqrt(0.5 * ((sigma_xx-sigma_yy)**2 + (sigma_yy-sigma_zz)**2 + (sigma_zz-sigma_xx)**2) + 
                            3 * (sigma_xy**2 + sigma_xz**2 + sigma_yz**2))

        ring_suffix = f"_ring{ring_index+1}"
        plot_and_save(sigma_xx, r'$\sigma_{xx}$', f"{map_name_pfx}_xx{ring_suffix}.png")
        plot_and_save(sigma_xy, r'$\sigma_{xy}$', f"{map_name_pfx}_xy{ring_suffix}.png")
        plot_and_save(sigma_yy, r'$\sigma_{yy}$', f"{map_name_pfx}_yy{ring_suffix}.png")
        plot_and_save(sigma_xz, r'$\sigma_{xz}$', f"{map_name_pfx}_xz{ring_suffix}.png")
        plot_and_save(sigma_yz, r'$\sigma_{yz}$', f"{map_name_pfx}_yz{ring_suffix}.png")
        plot_and_save(sigma_zz, r'$\sigma_{zz}$', f"{map_name_pfx}_zz{ring_suffix}.png")
        plot_and_save(vm_stress, r'$\sigma_{VM}$ (von Mises)', f"{map_name_pfx}_Mises{ring_suffix}.png")

    # Plot maps for each ring in parallel
    Parallel(n_jobs=-1)(delayed(_plot_one_ring)(i, filtered[i]) for i in range(num_rings))

    # --- Averaged Maps ---
    avg_s_xx = np.nanmean([np.array(r)[:,0].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_xy = np.nanmean([np.array(r)[:,1].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_yy = np.nanmean([np.array(r)[:,2].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_xz = np.nanmean([np.array(r)[:,3].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_yz = np.nanmean([np.array(r)[:,4].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_zz = np.nanmean([np.array(r)[:,5].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_s_vm = np.sqrt(0.5 * ((avg_s_xx-avg_s_yy)**2 + (avg_s_yy-avg_s_zz)**2 + (avg_s_zz-avg_s_xx)**2) + 
                       3 * (avg_s_xy**2 + avg_s_xz**2 + avg_s_yz**2))

    plot_and_save(avg_s_xx, r'$\sigma_{xx}$ (Avg)', f"{map_name_pfx}_xx_avg.png")
    plot_and_save(avg_s_xy, r'$\sigma_{xy}$ (Avg)', f"{map_name_pfx}_xy_avg.png")
    plot_and_save(avg_s_yy, r'$\sigma_{yy}$ (Avg)', f"{map_name_pfx}_yy_avg.png")
    plot_and_save(avg_s_xz, r'$\sigma_{xz}$ (Avg)', f"{map_name_pfx}_xz_avg.png")
    plot_and_save(avg_s_yz, r'$\sigma_{yz}$ (Avg)', f"{map_name_pfx}_yz_avg.png")
    plot_and_save(avg_s_zz, r'$\sigma_{zz}$ (Avg)', f"{map_name_pfx}_zz_avg.png")
    plot_and_save(avg_s_vm, r'$\sigma_{VM}$ (Avg)', f"{map_name_pfx}_Mises_avg.png")