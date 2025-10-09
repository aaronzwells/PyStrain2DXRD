import numpy as np
import matplotlib.pyplot as plt
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
    """Returns the filename without extension from a full path."""
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename.replace(".avg","")

# --- Utility: Creates directory for future output data storage
def create_directory(path, logger=None):
    """Creates a directory if it doesn't exist."""
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

    Args:
        x_data (np.ndarray): The dispersive axis data (e.g., q or 2-theta).
        y_data (np.ndarray): The intensity data.
    
    Returns:
        peak_positions (list): List of fitted centroid positions in the units of x_data.
    """
    logger = logger or logging.getLogger(__name__)

    # Data is now passed directly as arguments
    x = x_data
    y = y_data

    peaks, _ = find_peaks(y, height=np.max(y) * height_frac, distance=distance)
    if len(peaks) == 0:
        logger.warning("No peaks found in the provided data.")
        return [] # Return an empty list if no peaks are found

    dq = x[1] - x[0]
    widths_bins = peak_widths(y, peaks, rel_height=0.5)[0]
    widths_x = widths_bins * dq

    peak_positions = []
    for idx, wid in zip(peaks, widths_x):
        x0 = x[idx]
        half_width = max(2, int(np.ceil(wid / dq)))
        sl = slice(max(0, idx - half_width), min(len(x), idx + half_width + 1))
        try:
            p0 = [y[idx], x0, wid, eta0]
            bounds = ([0, x0 - delta_tol, 0, 0], [np.inf, x0 + delta_tol, np.inf, 1])
            # Note: Assumes pseudo_voigt function is defined elsewhere in the file
            popt, _ = curve_fit(pseudo_voigt, x[sl], y[sl], p0=p0, bounds=bounds)
            peak_positions.append(popt[1])
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
    
    Parameters:
        file_path (str): Path to the text file containing 2θ values (in degrees).
        wavelength_nm (float): Wavelength in nanometers.
    
    Returns:
        q_vals (np.ndarray): Array of q values in nm⁻¹.
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
    Calculates the mean and standard deviation of the pixels and sets the min 
    and max value based upon the a number of stdevs from the mean (k)
    """
    mean = np.mean(image)
    std = np.std(image)
    vmin = mean - k * std
    vmax = mean + k * std
    return exposure.rescale_intensity(image, in_range=(vmin, vmax))

# --- Pseudo-Voigt profile -----------------------------------------------
@njit(cache=True)
def pseudo_voigt(x, amp, cen, wid, eta):
    """ 
    standard definition of the Pseudo-Voigt function profile
    V(x,f) = η * L(x,f) + (1-η) * G(x,f), where 0 < η < 1
    ** η is defined by eta0 in the main pipeline
    Returns: V(x,f)
    """
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss   = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

# --- PyFAI data loading & integration -----------------------------------
def load_integrator_and_data(poni_path, tif_path, output_path, ref_tif_path=None, mask_threshold=4e2, logger=None):
    """
    Load/initialize PyFAI integrator and adjust a 2D XRD image using an auto-CB 
    scheme from ImageJ.
    Saves adjusted image as "<orig>_adjusted.tif".
    Returns: ai, data_adj, mask.
    """
    # Load calibrant and raw image
    ai  = pyFAI.load(poni_path)
    img = fabio.open(tif_path)
    data = img.data.astype(np.float32)

    # Contrast adjustment using ImageJ-style autocontrast (wider dynamic range)
    data_adj_float = imagej_autocontrast(data, k=3.0)

    # Keep float32 for PyFAI
    data_adj = data_adj_float

    logger = logger or logging.getLogger(__name__)
    # Save adjusted TIF alongside the original
    base, ext = os.path.splitext(tif_path)
    filename = os.path.basename(base)
    adjusted_path = f"{output_path}/{filename}_adjusted{ext}"
    imageio.imwrite(adjusted_path, data_adj)
    logger.info(f"Adjusted image saved to: {adjusted_path}")

    # No mask computation; return None for mask
    return ai, data_adj, None

def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000, output_dir=None, logger=None):
    """
    Use pyFAI to integrate the 2D pattern into azimuthal bins using pyFAI's integrate2d.
    Saves each azimuthal bin's q and intensity values as a .chi file (Fit2D style).

    Parameters:
        ai: pyFAI AzimuthalIntegrator object
        data: 2D diffraction pattern (numpy array)
        mask: mask for invalid pixels
        num_azim_bins: number of azimuthal bins (default: 360)
        q_min: minimum q value for radial integration
        npt_rad: number of radial points
        output_dir: directory where .chi files are saved

    Returns:
        I2d: 2D intensity array [azimuthal_bin x radial]
        q: radial q values
        chi: azimuthal chi values (degrees)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger = logger or logging.getLogger(__name__)
    logger.info("Running the integrate_2d() function")
    # Determine q_max from low-res integration
    q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial
    q_max = q_full[-1]

    # Perform integration
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

    # Transpose if necessary
    if I2d.shape == (len(q), len(chi)):
        I2d = I2d.T

    # Reorder chi to [0, 360) and rearrange I2d accordingly
    chi = (chi + 360) % 360
    order = np.argsort(chi)
    chi = chi[order]
    I2d = I2d[order]

    # Save each azimuthal bin's pattern as a .chi file
    for i, chi_val in enumerate(chi):
        chi_deg = chi_val  # * (num_azim_bins/360)
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

    Parameters:
        I2d (ndarray): 2D array of intensities [azimuthal_bin x radial]
        q (ndarray): Radial q values
        q_peaks (array-like): Initial peak positions to fit
        delta_tol (float): Fit range tolerance around initial guess
        eta0 (float): Initial pseudo-Voigt mixing parameter
        n_jobs (int): Number of parallel jobs
        delta_array (ndarray, optional): 2xN array where delta_array[0] gives tolerance up and
                                         delta_array[1] gives tolerance down for each ring.

    Returns:
        q_centroids (ndarray): Array of fitted centroid positions [num_peaks x num_azim_bins]
    """
    from joblib import Parallel, delayed
    logger = logger or logging.getLogger(__name__)

    dq = q[1] - q[0]
    widths_q = np.full_like(q_peaks, dq * 3)  # reasonable initial width
    # print(delta_array)
    def _fit_slice_w_guesses(intensity_row):
        out = []
        for i, (q0, wid0) in enumerate(zip(q_peaks, widths_q)):
            if delta_array is not None:
                tol_up = delta_array[0][i]
                tol_dn = delta_array[1][i]
            else:
                print("skipped delta_array")
                tol_up = tol_dn = delta_tol
            mask = (q >= q0 - tol_dn) & (q <= q0 + tol_up)
            x = q[mask]
            y = intensity_row[mask]
            if len(x) < 5 or not np.any(y > 0):
                out.append(np.nan)
                continue

            try:
                p0 = [np.max(y), q0, wid0, eta0]
                bounds = ([0, q0 - tol_dn, 0, 0], [np.inf, q0 + tol_up, np.inf, 1])
                popt, _ = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds)
                fitted_q = popt[1]
                
                # Reject if too close to edge or outside
                edge_buffer = 0  # small tolerance buffer (can be 0)
                if not (q0 - tol_dn + edge_buffer <= fitted_q <= q0 + tol_up - edge_buffer):
                    out.append(np.nan)
                else:
                    out.append(fitted_q)
            except Exception:
                logger.exception(f"Fit failed for peak index {i} in azimuthal bin.")
                out.append(np.nan)
        return out

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice_w_guesses)(row) for row in I2d
    )
    arr = np.array(results).T

    # Save q vs chi data to txt for future use
    if output_dir is not None:
        output_path = f"{output_dir}/q_vs_chi_peaks.txt"
        np.savetxt(output_path, arr, fmt="%.6f", delimiter="\t",
                header="Rows = diffraction rings; Columns = azimuthal bins (q vs chi data)")
        logger.info(f"q vs chi centroid data saved to: {output_path}")
    else:
        logger.warning("No output directory provided!\nq vs χ data was not saved to a .txt!")

    return arr, output_path


def plot_q_vs_chi_stacked(file_path, output_dir=None, chi_deg=None, dpi=600, plot=True, calibrant=False, logger=None):
    """
    Plots each row of q_vs_chi_peaks.txt as a stacked subplot, with chi on the x-axis and q on the y-axis.

    Parameters:
        file_path (str): Path to the q_vs_chi_peaks.txt file.
        output_path (str): Path to save the output plot.
        chi_deg (ndarray, optional): Azimuthal angle array in degrees. If None, assumes uniform [0, 360).
        dpi (int): Resolution of the saved figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    logger = logger or logging.getLogger(__name__)

    q_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    num_rings, num_chi = q_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, num_chi, endpoint=False)

    if plot:
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

    Parameters:
        file_path (str): Path to the strain_vs_chi_peaks.txt file.
        output_dir (str): Path to save the output plot.
        chi_deg (ndarray, optional): Azimuthal angle array in degrees. If None, assumes uniform [0, 360).
        dpi (int): Resolution of the saved figure.
    """
    import os
    logger = logger or logging.getLogger(__name__)
    strain_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    num_rings, num_chi = strain_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, num_chi, endpoint=False)

    if plot:
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
def fit_lattice_cone_distortion(q_data, q0_list, wavelength_nm,
                                chi_deg=None, psi_deg=None, phi_deg=None, omega_deg=None, num_strain_components=3, output_dir=None, dpi=600, plot=True, logger=None):
    """
    Fits lattice cone distortion model to q(chi) data to extract in-plane strain tensor components.

    Parameters:
        file_path (str): Path to q_vs_chi_peaks.txt
        output_dir (str): Where to save the combined plot and results
        calibrant (bool): True if using calibrant with predefined q0 values
        q0_vals (list or array): Known q0 values
        chi_deg (array-like, optional): Azimuthal angles in degrees. If None, assumes uniform [0, 360).
        dpi (int): Resolution for output plots
        plot (bool): Whether to generate and save plots
        logger (logging.Logger): Logger instance for logging information/warnings/errors

    Returns:
        strain_array (ndarray): Array of strain tensor components [eps_xx, eps_yy, eps_xy] per ring
        strain_list (list): Average strain per ring
        q0_list (list): q0 values per ring (optimized or predefined)
        strain_vs_chi_path (str): File path of saved strain vs chi data
    """
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
    axes = None
    if plot:
        fig, axes = plt.subplots(n_rings, 1, figsize=(10, 2 * n_rings), dpi=dpi, sharex=True)
        if n_rings == 1:
            axes = [axes]

    # y limits for the strain vs chi plots
    y_min = -0.0015
    y_max = 0.0015

    for i in range(n_rings):
        q_vals = q_data[i]
        mask = ~np.isnan(q_vals)
        x, y = chi_deg[mask], q_vals[mask]

        if len(x) < 20:
            logger.warning(f"Ring {i+1}: insufficient data points ({len(x)} < 20). Skipping fit.")
            if plot:
                ax = axes[i] if n_rings > 1 else axes[0]
                ax.set_title(f"Ring {i+1}: insufficient data")
                ax.axis('off')
            strain_params.append([np.nan]*7)
            # Append NaN fit for overlay plot as well
            fit_vs_chi.append(np.full(n_bins, np.nan))
            continue

        q0_fixed = q0_list[i]
        try:
            # Convert angles to radians for this azimuthal bin set
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
            # Intermediate parameters a, b, c from Table 1
            a = sin_theta * np.cos(omega_rad) + sin_chi * cos_theta * np.sin(omega_rad)
            b = -cos_chi * cos_theta
            c = sin_theta * np.sin(omega_rad) - sin_chi * cos_theta * np.cos(omega_rad)
            # Direction cosines A, B, C
            A = a * np.cos(phi_rad) - b * np.cos(psi_rad) * np.sin(phi_rad) + c * np.sin(psi_rad) * np.sin(phi_rad)
            B = a * np.sin(phi_rad) + b * np.cos(psi_rad) * np.cos(phi_rad) - c * np.sin(psi_rad) * np.cos(phi_rad)
            C = b * np.sin(psi_rad) + c * np.cos(psi_rad)
            
            epsilon = '\u03B5' # ε
            logger.info(f"The model is solving for {num_strain_components} strain components.")
            if num_strain_components == 6:
                # Strain coefficients f_ij
                f11 = A**2
                f12 = 2 * A * B
                f22 = B**2
                f13 = 2 * A * C
                f23 = 2 * B * C
                f33 = C**2
                if i==1: logger.info(f"No strain components are set to 0")
            elif num_strain_components == 5:
                f11 = A**2
                f12 = 2 * A * B
                f22 = B**2
                f13 = 2 * A * C
                f23 = 2 * B * C
                # out-of-plane normal strain zeroed as array
                f33 = np.zeros_like(A)
                if i==1: logger.info(f"{epsilon}33 is set to zero")
            elif num_strain_components == 3:
                f11 = A**2
                f12 = 2 * A * B
                f22 = B**2
                # out-of-plane strain components zeroed as array
                f13 = np.zeros_like(A)
                f23 = np.zeros_like(A)
                f33 = np.zeros_like(A)
                if i==1: logger.info(f"{epsilon}13, {epsilon}23, & {epsilon}33 are set to zero")
            else: 
                f11 = A**2
                f12 = 2 * A * B
                f22 = B**2
                f13 = np.zeros_like(A)
                f23 = np.zeros_like(A)
                f33 = np.zeros_like(A)
                if i==1: logger.warning(f"An incompatible number of strain components was selected. The only valid numbers are 3 (biaxial), 5 (biaxial w/ shear) & 6 (full strain tensor). Defaulting to biaxial.")

            # Build design matrix and measured y = ln(q0/q)
            F    = np.vstack([f11, f12, f22, f13, f23, f33]).T
            y_meas = np.log(q0_fixed / y)
            # Solve least squares
            eps, *_ = np.linalg.lstsq(F, y_meas, rcond=None)
            eps11, eps12, eps22, eps13, eps23, eps33 = eps
            strain_params.append([q0_fixed, eps11, eps12, eps22, eps13, eps23, eps33])
            # compute fitted strain vs chi using fixed θ0
            y_fit = F.dot(eps)
            # expand to full array
            fit_full = np.full(n_bins, np.nan)
            fit_full[mask] = y_fit
            fit_vs_chi.append(fit_full)
            if plot:
                ax = axes[i] if n_rings > 1 else axes[0]
                y_fit_lin = F.dot(eps)
                ax.plot(x, y_meas, '.', markersize=3, label='ln(q0/q)')
                ax.plot(x, y_fit_lin, '-', label=f'fit equation')
                ax.set_ylabel(f'{epsilon}=ln(q0/q)')
                ax.set_title(f'Ring {i+1}')
                ax.legend(fontsize='small', loc='lower left', bbox_to_anchor=(1.02,0.02))
        except Exception:
            if plot:
                ax = axes[i] if n_rings > 1 else axes[0]
                ax.set_title(f"Ring {i+1}: fit failed")
                ax.axis('off')
            strain_params.append([np.nan]*7)
            # Also append NaN fit for overlay plot
            fit_vs_chi.append(np.full(n_bins, np.nan))
            logger.exception(f"Fit failed for Ring {i+1}")

    if plot:
        axes[-1].set_xlabel('Azimuth χ (°)')
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "strain_vs_chi_plot_fitted.png")
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info(f"Combined distortion fit plot saved to: {fig_path}")

    # Convert strain_params to numpy array and save all six strain tensor components (excluding q0)
    strain_array = np.array([
        row[1:] if row is not None and len(row) == 7 
               else [np.nan] * 6
        for row in strain_params
    ])

    # Compute strain_list as described, with improved NaN handling
    strain_list = []
    for i, row in enumerate(q_data):
        mask = ~np.isnan(row)
        filtered_row = row[mask]
        q_avg = np.mean(filtered_row) if np.any(mask) else np.nan
        q0 = strain_params[i][0] if i < len(strain_params) and strain_params[i] is not None else np.nan
        if q0 != 0 and not np.isnan(q0) and not np.isnan(q_avg):
            strain = (q0 - q_avg) / q0
        else:
            strain = np.nan
        strain_list.append(strain)

    # Extract q0_list for all rings (already passed as argument, but keep for compatibility)
    q0_list_out = [row[0] if row is not None and len(row) > 0 else np.nan for row in strain_params]

    # Save the full strain vs chi array
    strain_vs_chi = (np.array(q0_list_out).reshape(-1, 1) - q_data) / np.array(q0_list_out).reshape(-1, 1)
    strain_vs_chi_path = os.path.join(output_dir, "strain_vs_chi_peaks.txt")
    np.savetxt(strain_vs_chi_path, strain_vs_chi, fmt="%.6e", delimiter="\t",
               header="Rows = diffraction rings; Columns = azimuthal bins (strain vs chi data)")
    logger.info(f"Strain vs chi centroid data saved to: {strain_vs_chi_path}")

    return strain_array, strain_list, q0_list_out, strain_vs_chi_path

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
    colorbar_scale=None,
    output_dir="StrainMaps",
    dpi=600,
    map_name_pfx="strain-map_",
    logger=None,
):
    """
    Generates physically accurate strain maps by drawing individual rectangles for each data point.
    Includes options for coordinate shifting and trimming negative axes.
    """
    import os
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

    num_points = len(strain_data)
    if num_points != n_rows * n_cols:
        raise ValueError(f"Grid dimension mismatch! JSON data points ({num_points}) do not match grid ({n_rows}x{n_cols}).")
    logger.info(f"Loaded {num_points} data points for a {n_rows}x{n_cols} grid.")

    num_rings = len(strain_data[0].get("strain_tensor", []))
    if gap_mm is None: gap_mm = 0.0

    filtered = [[] for _ in range(num_rings)]
    # (This data filtering loop is unchanged)
    for entry in strain_data:
        tensors = entry.get("strain_tensor", [])
        for i in range(num_rings):
            if i < len(tensors) and isinstance(tensors[i], dict):
                filtered[i].append([
                    tensors[i].get("eps_xx", np.nan), tensors[i].get("eps_xy", np.nan),
                    tensors[i].get("eps_yy", np.nan), tensors[i].get("eps_xz", np.nan),
                    tensors[i].get("eps_yz", np.nan), tensors[i].get("eps_zz", np.nan)
                ])
            else:
                filtered[i].append([np.nan] * 6)

    pixel_size_unit = "mm"
    
    # --- Apply the coordinate offset ---
    startX, startY = start_xy
    shiftX, shiftY = map_offset_xy
    startX += shiftX
    startY += shiftY
    
    dX, dY = step_size

    def plot_and_save(data, title, filename):
        # Flipping the data array for plotting
        data = np.flipud(data)

        fig, ax = plt.subplots(figsize=(3.5, 4), dpi=dpi)
        cmap = plt.cm.jet.copy()
        
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if color_limit_window:
            x_min_win, x_max_win = color_limit_window
            # Adjust window by the same offset to sample the correct region
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

        # --- Calculate map edges and apply trimming ---
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
        
        # Colorbar and label definitions
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cb.set_label('Strain')
        cb.formatter = FuncFormatter(lambda x, _: f"{x:.1e}")
        cb.update_ticks()

        ax.set_title(title)
        ax.set_xlabel(f'X Position [{pixel_size_unit}]')
        ax.set_ylabel(f'Y Position [{pixel_size_unit}]')
        plt.tight_layout()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"{title} heatmap saved to: {filepath}")

    def _plot_one_ring(ring_index, ring_data):
        strain_array = np.array(ring_data).reshape((n_rows, n_cols, 6))
        
        eps_xx = strain_array[:, :, 0]
        eps_xy = strain_array[:, :, 1]
        eps_yy = strain_array[:, :, 2]
        eps_xz = strain_array[:, :, 3]
        eps_yz = strain_array[:, :, 4]
        eps_zz = strain_array[:, :, 5]
        vm_base = np.sqrt(((eps_xx-eps_yy)**2 + (eps_yy-eps_zz)**2 + (eps_zz-eps_xx)**2)/2 + 3*(eps_xy**2 + eps_xz**2 + eps_yz**2))

        ring_suffix = f"_ring{ring_index+1}"
        plot_and_save(eps_xx, r'$\varepsilon_{xx}$', f"{map_name_pfx}_xx{ring_suffix}.png")
        plot_and_save(eps_xy, r'$\varepsilon_{xy}$', f"{map_name_pfx}_xy{ring_suffix}.png")
        plot_and_save(eps_yy, r'$\varepsilon_{yy}$', f"{map_name_pfx}_yy{ring_suffix}.png")
        plot_and_save(eps_xz, r'$\varepsilon_{xz}$', f"{map_name_pfx}_xz{ring_suffix}.png")
        plot_and_save(eps_yz, r'$\varepsilon_{yz}$', f"{map_name_pfx}_yz{ring_suffix}.png")
        plot_and_save(eps_zz, r'$\varepsilon_{zz}$', f"{map_name_pfx}_zz{ring_suffix}.png")
        plot_and_save(vm_base, r'$\varepsilon_{VM}$', f"{map_name_pfx}_Mises{ring_suffix}.png")

    Parallel(n_jobs=-1)(delayed(_plot_one_ring)(i, filtered[i]) for i in range(num_rings))

    # Averaged maps
    avg_eps_xx = np.nanmean([np.array(r)[:,0].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_xy = np.nanmean([np.array(r)[:,1].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_yy = np.nanmean([np.array(r)[:,2].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_xz = np.nanmean([np.array(r)[:,3].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_yz = np.nanmean([np.array(r)[:,4].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_zz = np.nanmean([np.array(r)[:,5].reshape(n_rows,n_cols) for r in filtered], axis=0)
    avg_eps_vm = np.sqrt(((avg_eps_xx-avg_eps_yy)**2 + (avg_eps_yy-avg_eps_zz)**2 + (avg_eps_zz-avg_eps_xx)**2)/2 + 3*(avg_eps_xy**2+avg_eps_xz**2+avg_eps_yz**2))

    plot_and_save(avg_eps_xx, r'$\varepsilon_{xx}$ (Avg)', f"{map_name_pfx}_xx_avg.png")
    plot_and_save(avg_eps_xy, r'$\varepsilon_{xy}$ (Avg)', f"{map_name_pfx}_xy_avg.png")
    plot_and_save(avg_eps_yy, r'$\varepsilon_{yy}$ (Avg)', f"{map_name_pfx}_yy_avg.png")
    plot_and_save(avg_eps_xz, r'$\varepsilon_{xz}$ (Avg)', f"{map_name_pfx}_xz_avg.png")
    plot_and_save(avg_eps_yz, r'$\varepsilon_{yz}$ (Avg)', f"{map_name_pfx}_yz_avg.png")
    plot_and_save(avg_eps_zz, r'$\varepsilon_{zz}$ (Avg)', f"{map_name_pfx}_zz_avg.png")
    plot_and_save(avg_eps_vm, r'$\varepsilon_{VM}$ (Avg)', f"{map_name_pfx}_Mises_avg.png")

# --- Utility: Reconstruct simulated diffraction rings using fitted strain tensor components ---
def reconstruct_rings_from_json(json_path, wavelength_nm, chi_step=1.0, logger=None, plot=True, output_dir=None):
    """
    Reconstruct simulated diffraction rings using fitted strain tensor components.

    Parameters:
        json_path (str): Path to strain_tensor_summary.json
        wavelength_nm (float): X-ray wavelength (nm)
        chi_step (float): Step size in degrees for χ grid
        logger (logging.Logger): Optional logger instance
        plot (bool): Whether to generate plots
        output_dir (str): Directory to save plots

    Returns:
        dict: {ring_index: (chi_deg, q_sim)}
    """

    with open(json_path, "r") as f:
        strain_data = json.load(f)

    chi_deg = np.arange(0, 360, chi_step)
    # Apply the same convention correction as in fit_lattice_cone_distortion
    chi_rad = np.deg2rad(90.0 - chi_deg)
    results = {}

    for entry in strain_data:
        if "strain_tensor" not in entry:
            continue
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
    """
    import os
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
    
    # --- Strain to Stress Conversion ---
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
            
            if np.isnan(eps_xx):
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

    # The rest of the function is similar to the strain mapping function
    num_points = len(stress_data)
    if num_points != n_rows * n_cols:
        raise ValueError(f"Grid dimension mismatch! JSON data points ({num_points}) do not match grid ({n_rows}x{n_cols}).")
    logger.info(f"Calculated stress for {num_points} points on a {n_rows}x{n_cols} grid.")

    num_rings = len(stress_data[0]) if stress_data else 0
    if gap_mm is None: gap_mm = 0.0

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

    Parallel(n_jobs=-1)(delayed(_plot_one_ring)(i, filtered[i]) for i in range(num_rings))

    # Averaged maps
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