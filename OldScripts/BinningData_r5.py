import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import pyFAI, fabio

def pseudo_voigt(x, amp, cen, wid, eta):
    """
    Pseudo-Voigt profile:
      - amp: amplitude
      - cen: center (q) position
      - wid: full-width at half-maximum (FWHM)
      - eta: Lorentzian fraction (0: Gaussian, 1: Lorentzian)
    """
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

def load_integrator_and_data(poni_path, tif_path, mask_threshold=4e9):
    """Load a PyFAI integrator and TIFF data, returning (ai, data, mask)."""
    ai = pyFAI.load(poni_path)
    img = fabio.open(tif_path)
    data = img.data
    mask = data > mask_threshold
    return ai, data, mask

def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000):
    """
    Perform a 2D integration over q ≥ q_min (in nm⁻¹), returning
    I2d (shape [n_azim, n_rad]), q (radial coords), and chi (azimuthal coords).
    """
    # get full radial axis
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
    # ensure shape is [n_azim, n_rad]
    if I2d.shape == (len(q), len(chi)):
        I2d = I2d.T
    return I2d, q, chi

def detect_global_peaks(I2d, q, num_rings=8, height_frac=0.09, distance=20,
                        delta_tol=0.05, eta0=0.5):
    """
    Detect global peak positions in the *mean* radial profile, refine them
    via pseudo-Voigt fits, and return:
      q_peaks      : array of refined q-centers (length num_rings)
      peak_windows : list of slice objects for each ring
      widths_q     : initial FWHM guesses in q-units
    """
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

def _fit_slice(int_row, q_peaks, peak_windows, widths_q, q, delta_tol=0.05, eta0=0.5):
    """Helper: fit all rings in a single chi‐slice intensity row."""
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

def fit_slices_parallel(I2d, q_peaks, peak_windows, widths_q, q, n_jobs=-1, delta_tol=0.05, eta0=0.5):
    """
    Fit every azimuthal slice in parallel. Returns q_vs_chi with shape
    (n_rings, n_chi).
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice)(row, q_peaks, peak_windows, widths_q, q, delta_tol, eta0)
        for row in I2d
    )
    q_vs_chi = np.array(results).T
    return q_vs_chi

def plot_q_vs_chi(chi, q_vs_chi, q_peaks, delta_tol=0.05,
                  xlabel='Azimuth (°)', output_path=None):
    """
    Make a stacked plot of q_vs_chi[j] vs chi for each ring j.
    If output_path is given, save the figure there.
    """
    n_rings = len(q_peaks)
    fig, axes = plt.subplots(n_rings, 1, sharex=True, figsize=(6, 2*n_rings))
    for j, ax in enumerate(axes):
        ax.plot(chi, q_vs_chi[j], '-k', lw=1)
        m = np.nanmean(q_vs_chi[j])
        ax.set_ylim(m - delta_tol, m + delta_tol)
        ax.set_ylabel(f"q ≃ {q_peaks[j]:.2f}")
        ax.set_xlim(chi.min(), chi.max())
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.show()

def main():
    # --- user‐set parameters ---
    poni_file      = "calibration/Calibration_LaB6_100x100_3s_r8.poni"
    tif_file       = "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif"
    mask_thresh    = 4e9
    num_azim_bins  = 120
    q_min_nm1      = 16.0
    npt_rad        = 5000
    delta_tol      = 0.05
    eta0           = 0.5
    output_png     = f"OutputFiles/QvsAz/SSAO-6_25C_Map_100x100_{num_azim_bins}bin_test.png"

    # --- pipeline ---
    ai, data, mask = load_integrator_and_data(poni_file, tif_file, mask_thresh)
    I2d, q, chi    = integrate_2d(ai, data, mask,
                                  num_azim_bins, q_min_nm1, npt_rad)
    q_peaks, pw, wq = detect_global_peaks(I2d, q, delta_tol=delta_tol, eta0=eta0)
    q_vs_chi       = fit_slices_parallel(I2d, q_peaks, pw, wq, q,
                                         delta_tol=delta_tol, eta0=eta0)
    plot_q_vs_chi(chi, q_vs_chi, q_peaks, delta_tol,
                  xlabel='Azimuth (°)', output_path=output_png)

if __name__ == "__main__":
    main()
