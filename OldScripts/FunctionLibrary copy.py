import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from skimage import exposure
import os
import imageio.v2 as imageio
import pyFAI, fabio
import warnings

# --- Utility: Removes the extension from a file name
def remove_filename_extension(path):
    """Returns the filename without extension from a full path."""
    filename = os.path.splitext(os.path.basename(path))[0]
    return filename.replace(".avg","")

# --- Utility: Creates directory for future output data storage
def create_directory(path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory created: {path}")
    except Exception as e:
        warnings.warn(f"[WARNING] Error creating directory: {e}",UserWarning)
    return path

# --- Utility: Fit pseudo-Voigt to peaks in a .int file ------------------
# .int file is generated as from an ideal alumina (corundum) crystal structure
# from materialsproject.org database (mp-1143) and Vesta to simulate the structure
def validate_curve_fitting(int_file_path, height_frac=0.1, distance=20, eta0=0.5, delta_tol=0.1):
    """
    Reads a .int file with columns: 2theta, intensity, (ignored third column),
    and fits pseudo-Voigt profiles to detect peak centroids.

    Returns:
        peak_positions: List of centroid positions in 2theta.
    """
    import csv

    data = np.loadtxt(int_file_path, comments='#')
    if data.shape[1] < 2:
        raise ValueError("Expected at least two columns in .int file.")
    x = data[:, 0]
    y = data[:, 1]

    peaks, _ = find_peaks(y, height=np.max(y) * height_frac, distance=distance)
    if len(peaks) == 0:
        raise ValueError("No peaks found in .int file.")

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
            popt, _ = curve_fit(pseudo_voigt, x[sl], y[sl], p0=p0, bounds=bounds)
            peak_positions.append(popt[1])
        except Exception as e:
            warnings.warn(f"[WARNING] Fit failed at index {idx} with x0={x0:.2f}: {e}", UserWarning)
            continue
    outputFilePath = "AdditionalFiles/FxnValidation/FitPeakLocations-Al2O3.txt"
    with open(outputFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        for pos in peak_positions:
            writer.writerow([pos])
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
def load_integrator_and_data(poni_path, tif_path, output_path,ref_tif_path=None, mask_threshold=4e2):
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

    # Save adjusted TIF alongside the original
    base, ext = os.path.splitext(tif_path)
    filename = os.path.basename(base)
    adjusted_path = f"{output_path}/{filename}_adjusted{ext}"
    imageio.imwrite(adjusted_path, data_adj)
    print(f"[INFO] Adjusted image saved to: {adjusted_path}")

    # No mask computation; return None for mask
    return ai, data_adj, None

def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000, output_dir=None):
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

    print("[INFO] Running the integrate_2d() function")
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
        # print(f"Chi File Production #: {i}")
        chi_deg = chi_val# * (num_azim_bins/360)
        filename = os.path.join(output_dir, f"azim_{int(round(chi_deg))}deg.chi")
        with open(filename, 'w') as f:
            f.write(f"# Azimuthal bin: {chi_deg:.2f} deg\n")
            f.write("# Columns: q (nm^-1), Intensity (a.u.)\n")
            for q_val, I_val in zip(q, I2d[i]):
                f.write(f"{q_val:.6f} {I_val:.6f}\n")
        # print(f"Chi file created: azim_{int(round(chi_deg))}deg.chi")

    return I2d, q, chi

def fit_peaks_with_initial_guesses(I2d, q, q_peaks, delta_tol=0.07, eta0=0.5, n_jobs=-1, delta_array=None, output_dir=None):
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
                out.append(np.nan)
        return out

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice_w_guesses)(row) for row in I2d
    )
    arr = np.array(results).T

    # Save q vs chi data to txt for future use
    if output_dir != None:
        output_path = f"{output_dir}/q_vs_chi_peaks.txt"
        np.savetxt(output_path, arr, fmt="%.6f", delimiter="\t",
                header="Rows = diffraction rings; Columns = azimuthal bins (q vs chi data)")
        print(f"[INFO] q vs chi centroid data saved to: {output_path}")
    else:
        warnings.warn("[WARNING] No output directory provided!\nq vs χ data was not saved to a .txt!", UserWarning)

    return arr, output_path

def plot_q_vs_chi_stacked(file_path, output_dir=None, chi_deg=None, dpi=600):
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

    q_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    num_rings, num_chi = q_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, num_chi, endpoint=False)

    fig, axes = plt.subplots(num_rings, 1, figsize=(8, 2 * num_rings), sharex=True, dpi=dpi)

    for i in range(num_rings):
        ax = axes[i] if num_rings > 1 else axes
        q_vals = q_data[i]
        mask = ~np.isnan(q_vals)
        ax.plot(chi_deg[mask], q_vals[mask], '.', markersize=3)
        ax.set_title(f'Ring {i+1}')
        ax.set_ylabel(f'q (nm⁻¹)')
        ax.set_xlim(0, 360)
        # ax.grid(True)

    axes[-1].set_xlabel('Azimuth χ (°)')
    fig.tight_layout()
    fig_filename = os.path.join(output_dir, "q_vs_chi_plot.png")
    fig.savefig(fig_filename)
    plt.close(fig)
    print(f"[INFO] Stacked q vs chi plot saved to: {fig_filename}")

# --- Compute full strain tensor ----------------------------------------
def fit_lattice_cone_distortion(file_path, output_dir=None, chi_deg=None, dpi=600):
    """
    Fits lattice cone distortion model to q(chi) data to extract in-plane strain tensor components.

    Parameters:
        file_path (str): Path to q_vs_chi_peaks.txt
        output_dir (str): Where to save the combined plot and results
        chi_deg (array-like, optional): Azimuthal angles in degrees. If None, assumes uniform [0, 360).

    Returns:
        strain_params (ndarray): Array of [eps_xx, eps_yy, eps_xy] per ring
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    q_data = np.loadtxt(file_path, comments='#', delimiter='\t')
    n_rings, n_bins = q_data.shape

    if chi_deg is None:
        chi_deg = np.linspace(0, 360, n_bins, endpoint=False)

    def distortion_model(chi_deg, q0, eps_xx, eps_yy, eps_xy):
        chi_rad = np.deg2rad(chi_deg)
        eps = eps_xx * np.cos(chi_rad)**2 + eps_yy * np.sin(chi_rad)**2 + eps_xy * np.sin(2 * chi_rad)
        return q0 * (1 - eps)

    strain_params = []
    fig, axes = plt.subplots(n_rings, 1, figsize=(10, 2 * n_rings), dpi=dpi, sharex=True)
    if n_rings == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        q_vals = q_data[i]
        mask = ~np.isnan(q_vals)
        x, y = chi_deg[mask], q_vals[mask]

        if len(x) < 5:
            ax.set_title(f"Ring {i+1}: insufficient data")
            ax.axis('off')
            strain_params.append([np.nan]*3)
            continue

        p0 = [np.mean(y), 0, 0, 0]
        try:
            popt, _ = curve_fit(distortion_model, x, y, p0=p0)
            q0, eps_xx, eps_yy, eps_xy = popt
            y_fit = distortion_model(x, *popt)
            # Compute residuals standard deviation
            residuals_std = np.std(y - y_fit)
            strain_params.append([q0, eps_xx, eps_yy, eps_xy])
            ax.plot(x, y, '.', markersize=3, label='Centroid Data')
            ax.plot(
                x, y_fit, '-',
                label=f'ε_xx={eps_xx:.3e}, \nε_yy={eps_yy:.3e}, \nε_xy={eps_xy:.3e} \nStd(res)={residuals_std:.2e}'
            )
            # ax.set_xlabel('Azimuth χ (°)')
            ax.set_ylabel('q (nm⁻¹)')
            ax.set_title(f'Ring {i+1}')
            ax.legend(fontsize='small')
            ax.legend(loc='lower left', bbox_to_anchor=(1.02,0.02), ncol=1)
        except Exception as e:
            ax.set_title(f"Ring {i+1}: fit failed")
            ax.axis('off')
            strain_params.append([np.nan, np.nan, np.nan, np.nan])
            warnings.warn(f"[WARNING] Fit failed for Ring {i+1}: {e}", UserWarning)

    axes[-1].set_xlabel('Azimuth χ (°)')
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "q_vs_chi_plot_fitted.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[INFO] Combined distortion fit plot saved to: {fig_path}")

    # Convert strain_params to numpy array and save (only the tensor components, not q0)
    strain_array = np.array([row[1:4] if row is not None and len(row) > 3 else [np.nan, np.nan, np.nan] for row in strain_params])
    out_txt = os.path.join(output_dir, "strain_tensor_components.txt")
    np.savetxt(out_txt, strain_array, header="Columns: eps_xx eps_yy eps_xy", fmt="%.6e", delimiter="\t")
    print(f"[INFO] Strain tensor components saved to: {out_txt}")

    # Compute strain_list as described
    strain_list = []
    for i, row in enumerate(q_data):
        mask = ~np.isnan(row)
        q_avg = np.mean(row[mask]) if np.any(mask) else np.nan
        q0 = strain_params[i][0] if i < len(strain_params) and strain_params[i] is not None else np.nan
        strain = (q0 - q_avg) / q0 if q0 != 0 and not np.isnan(q0) else np.nan
        strain_list.append(strain)

    return strain_array, strain_list

import numpy as np

def compute_bulk_strain_stress_xray(strain_list, hkl_list, E, nu):
    """
    Computes bulk stress and strain using plane-specific X-ray elastic constants and strain data.

    Parameters:
    -----------
    strain_list : list of float
        Measured strains ε_(hkl) for each ring in increasing q order.
    hkl_list : list of tuple
        Miller indices (h,k,l) for each ring.
    E : float
        Young's modulus of the material [Pa].
    nu : float
        Poisson's ratio of the material.

    Returns:
    --------
    bulk_stress : float
        Bulk stress [Pa].
    bulk_strain : float
        Bulk strain (unitless).
    stress_list : list of float
        Stress for each (hkl) plane [Pa].
    """

    S1 = (1 + nu) / E
    S2_half = (1 - 2 * nu) / E

    gamma_list = []
    stress_list = []
    weights = []

    for eps, hkl in zip(strain_list, hkl_list):
        h, k, l = hkl
        h2, k2, l2 = h**2, k**2, l**2
        denom = (h2 + k2 + l2)**2
        gamma = (h2 * k2 + h2 * l2 + k2 * l2) / denom
        gamma_list.append(gamma)

        denom_factor = S1 + S2_half * gamma
        sigma = eps / denom_factor
        stress_list.append(sigma)

        weight = 1 / (1 + (S2_half * gamma / S1))
        weights.append(weight)

    weights = np.array(weights)
    stress_array = np.array(stress_list)

    bulk_stress = np.sum(weights * stress_array) / np.sum(weights)
    bulk_strain = S1 * bulk_stress

    return bulk_stress, bulk_strain, stress_list