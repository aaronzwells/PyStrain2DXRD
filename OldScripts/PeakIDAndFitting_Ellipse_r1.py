import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import pyFAI, fabio
from skimage.measure import EllipseModel

# --- original pseudo-Voigt peak routine ---
def pseudo_voigt(x, amp, cen, wid, eta):
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

# --- data loading and integration (unchanged) ---
def load_integrator_and_data(poni_path, tif_path, mask_threshold=4e9):
    ai = pyFAI.load(poni_path)
    img = fabio.open(tif_path)
    data = img.data
    mask = data > mask_threshold
    return ai, data, mask


def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000):
    q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial
    q_max = q_full[-1]
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
    if I2d.shape == (len(q), len(chi)):
        I2d = I2d.T
    return I2d, q, chi

# --- peak detection (unchanged) ---
def detect_global_peaks(I2d, q, num_rings=8, height_frac=0.09, distance=20,
                        delta_tol=0.05, eta0=0.5):
    radial_mean = I2d.mean(axis=0)
    init_inds, _ = find_peaks(
        radial_mean,
        height=radial_mean.max() * height_frac,
        distance=distance
    )
    widths_bins = peak_widths(radial_mean, init_inds, rel_height=0.5)[0]
    q_initials  = q[init_inds]
    sel = np.argsort(q_initials)[:num_rings]
    init_inds, q_initials, widths_bins = init_inds[sel], q_initials[sel], widths_bins[sel]
    widths_q = widths_bins * (q[1] - q[0])
    q_peaks = []
    peak_windows = []
    half_bin_factor = (q[1] - q[0])
    for idx0, q0, wid0 in zip(init_inds, q_initials, widths_q):
        half_bins = int(np.ceil(wid0 / half_bin_factor))
        wl = slice(max(0, idx0-half_bins), min(len(q), idx0+half_bins+1))
        x, y = q[wl], radial_mean[wl]
        try:
            p0 = [y.max(), q0, wid0, eta0]
            bounds = ([0, q0-delta_tol, 0, 0], [np.inf, q0+delta_tol, np.inf, 1])
            popt, _ = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds)
            qc = popt[1]
        except Exception:
            qc = q0
        q_peaks.append(qc)
        i_cen = np.argmin(np.abs(q - qc))
        bw = max(2, half_bins)
        peak_windows.append(slice(i_cen-bw, i_cen+bw+1))
    return np.array(q_peaks), peak_windows, widths_q

# --- fitting slices in parallel (unchanged) ---
def _fit_slice(int_row, q_peaks, peak_windows, widths_q, q, delta_tol=0.05, eta0=0.5):
    out = []
    for wl, q0, wid0 in zip(peak_windows, q_peaks, widths_q):
        x, y = q[wl], int_row[wl]
        if len(x) < 5:
            out.append(np.nan)
            continue
        p0 = [y.max(), q0, wid0, eta0]
        bounds = ([0, q0-delta_tol, 0, 0], [np.inf, q0+delta_tol, np.inf, 1])
        try:
            popt, _ = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds, maxfev=1000)
            out.append(popt[1])
        except Exception:
            out.append(np.nan)
    return out


def fit_slices_parallel(I2d, q_peaks, peak_windows, widths_q, q, n_jobs=-1,
                         delta_tol=0.05, eta0=0.5):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice)(row, q_peaks, peak_windows, widths_q, q, delta_tol, eta0)
        for row in I2d
    )
    return np.array(results).T

# --- new plotting using EllipseModel (0°–360°) ---

def plot_ellipse_fits(chi, q_vs_chi, output_path=None, dpi=600):
    """
    Plot raw q_vs_chi data (0°–360°) and overlay an ellipse fit for each ring.
    """
    # remap chi from [-180,180] to [0,360)
    chi_mod = np.mod(chi, 360)
    sort_idx = np.argsort(chi_mod)
    chi_plot = chi_mod[sort_idx]
    phi = np.deg2rad(chi_plot)

    n_rings, _ = q_vs_chi.shape
    fig, axes = plt.subplots(n_rings, 1, sharex=True,
                             figsize=(6, 2*n_rings), dpi=dpi)
    for j, ax in enumerate(axes):
        r_vals_all = q_vs_chi[j]
        r_vals = r_vals_all[sort_idx]
        mask = ~np.isnan(r_vals)
        if not mask.any():
            continue
        phi_j = phi[mask]
        r_j   = r_vals[mask]
        x = r_j * np.cos(phi_j)
        y = r_j * np.sin(phi_j)
        pts = np.column_stack([x, y])
        model = EllipseModel()
        model.estimate(pts)
        xc, yc, a, b, theta = model.params
        # generate fitted ellipse
        t = np.linspace(0, 2*np.pi, 200)
        xt = xc + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
        yt = yc + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
        rt = np.sqrt(xt**2 + yt**2)
        chi_fit = np.rad2deg(t)
        # plot
        ax.plot(chi_plot, r_vals, '.', ms=2, label='data')
        ax.plot(chi_fit, rt, '-', lw=1, label='ellipse fit')
        ax.set_ylabel(f"Ring {j+1}")
        ax.legend(loc='upper right', fontsize='small')
    axes[-1].set_xlabel('Azimuth (°)')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi)
    plt.show()

# --- main pipeline ---

def main():
    # user parameters
    poni_file   = "calibration/Calibration_LaB6_100x100_3s_r8.poni"
    tif_file    = "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif"
    mask_thresh = 4e9
    num_azim    = 120
    q_min_nm1   = 16.0
    npt_rad     = 5000
    output_png  = f"OutputFiles/Fits/SSAO-6_25C_Map_ellipse_{num_azim}bin.png"

    # load & integrate
    ai, data, mask   = load_integrator_and_data(poni_file, tif_file, mask_thresh)
    I2d, q, chi      = integrate_2d(ai, data, mask, num_azim, q_min_nm1, npt_rad)
    # detect and fit peaks
    q_peaks, pw, wq  = detect_global_peaks(I2d, q)
    # fit azimuthal slices
    q_vs_chi         = fit_slices_parallel(I2d, q_peaks, pw, wq, q)
    # plot and save results
    plot_ellipse_fits(chi, q_vs_chi, output_path=output_png)

if __name__ == "__main__":
    main()
