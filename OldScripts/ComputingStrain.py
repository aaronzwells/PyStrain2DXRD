import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from skimage import exposure
import os
import imageio
import pyFAI, fabio

# --- Pseudo-Voigt profile -----------------------------------------------
def pseudo_voigt(x, amp, cen, wid, eta):
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss   = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

# --- PyFAI data loading & integration -----------------------------------
def load_integrator_and_data(poni_path, tif_path, mask_threshold=4e9):
    ai = pyFAI.load(poni_path)
    img = fabio.open(tif_path)
    data = img.data
    mask = data > mask_threshold
    return ai, data, mask


def integrate_2d(ai, data, mask, num_azim_bins=360, q_min=16.0, npt_rad=5000):
    q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial
    q_max  = q_full[-1]
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

# --- Global peak detection & window slicing -----------------------------
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
    half_bin = q[1] - q[0]

    for idx0, q0, wid0 in zip(init_inds, q_initials, widths_q):
        half_bins = int(np.ceil(wid0 / half_bin))
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

# --- Slice-by-slice Pseudo-Voigt centroid fitting ----------------------
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

def fit_slices_parallel(I2d, q_peaks, peak_windows, widths_q, q,
                        n_jobs=-1, delta_tol=0.05, eta0=0.5):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_slice)(row, q_peaks, peak_windows, widths_q, q, delta_tol, eta0)
        for row in I2d
    )
    return np.array(results).T

# --- Fourier fit helper -------------------------------------------------
def fit_fourier(r_vals, phi_vals, harmonics):
    cols = [np.ones_like(phi_vals)]
    for n in harmonics:
        cols.append(np.cos(n * phi_vals))
        cols.append(np.sin(n * phi_vals))
    A = np.vstack(cols).T
    coeffs, *_ = np.linalg.lstsq(A, r_vals, rcond=None)
    r_fit = A.dot(coeffs)
    return coeffs, r_fit

# --- Compute full strain tensor -----------------------------------------
def compute_strain_tensor(q_vs_chi, chi_deg, integrator, harmonics):
    eqs, rhs = [], []
    phi = np.deg2rad(chi_deg)
    # wavelength in nm
    lam_nm = integrator.wavelength * 1e9
    for j in range(q_vs_chi.shape[0]):
        r_vals = q_vs_chi[j]
        if np.all(np.isnan(r_vals)):
            continue
        a0 = np.nanmean(r_vals)
        eps_vals = (r_vals - a0) / a0
        coeffs, _ = fit_fourier(eps_vals, phi, harmonics)
        c0, c1, s1, c2, s2 = coeffs
        sin_theta = lam_nm * a0 / (4 * np.pi)
        cos_theta = np.sqrt(np.clip(1 - sin_theta**2, 0, 1))
        sin_psi = cos_theta
        cos_psi = sin_theta
        L2 = sin_psi**2
        L1 = sin_psi * cos_psi
        # Build equations per ring
        eqs += [[L2/2, -L2/2, 0,      0,    0,    0]]; rhs.append(c2)
        eqs += [[0,     0,      0,      L2,  0,    0]]; rhs.append(s2)
        eqs += [[0,     0,      0,      0,    L1,  0]]; rhs.append(c1)
        eqs += [[0,     0,      0,      0,    0,    L1]]; rhs.append(s1)
        eqs += [[L2/2,  L2/2,   cos_psi**2, 0, 0, 0]]; rhs.append(c0)
    A = np.array(eqs)
    b = np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    eps11, eps22, eps33, eps12, eps13, eps23 = sol
    return np.array([[eps11, eps12, eps13],
                     [eps12, eps22, eps23],
                     [eps13, eps23, eps33]])

# --- Q vs Azimuth stacked plot ------------------------------------------
def plot_fitted_q_vs_chi(chi, q_vs_chi, delta_tol, harmonics,
                         output_path=None, dpi=600):
    n_rings = q_vs_chi.shape[0]
    phi = np.deg2rad(chi)
    fig, axes = plt.subplots(n_rings, 1, sharex=True,
                             figsize=(6, 2*n_rings), dpi=dpi)
    for j, ax in enumerate(axes):
        r_vals = q_vs_chi[j]
        if np.all(np.isnan(r_vals)):
            continue
        _, r_fit = fit_fourier(r_vals, phi, harmonics)
        ax.plot(chi, r_vals, '.', ms=1, label='Centroids')
        ax.plot(chi, r_fit, '-', lw=1, label='Fourier fit')
        m = np.nanmean(r_vals)
        ax.set_ylim(m - delta_tol, m + delta_tol)
        ax.set_xlim(0, 360)
        ax.set_ylabel(f"Ring {j+1}")
        ax.legend(fontsize='small')
    axes[-1].set_xlabel('Azimuth (°)')
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi)

# --- Radar (polar) plot for all rings -----------------------------------
def plot_radar_all_rings(chi, q_vs_chi, harmonics, output_path=None, dpi=600):
    phi = np.deg2rad(chi)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},
                           figsize=(6,6), dpi=dpi)
    for j in range(q_vs_chi.shape[0]):
        r_vals = q_vs_chi[j]
        if np.all(np.isnan(r_vals)):
            continue
        _, r_fit = fit_fourier(r_vals, phi, harmonics)
        ax.plot(phi, r_vals, '.', ms=3, label=f'Ring {j+1}')
        ax.plot(phi, r_fit, '-', lw=1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Radar: q vs Azimuth')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')

# --- Main pipeline -------------------------------------------------------
def main():
    poni_file     = "calibration/Calibration_LaB6_100x100_3s_r8.poni"
    tif_file      = "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif"
    mask_thresh   = 4e9
    num_azim_bins = 120
    q_min_nm1     = 16.0
    npt_rad       = 5000
    delta_tol     = 0.05
    eta0          = 0.5
    harmonics     = (1, 2)
    qazi_png       = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_fourier_{num_azim_bins}bin.png"
    radar_png      = f"OutputFiles/Fits/SSAO-6_25C_Map_100x100_radar_{num_azim_bins}bin.png"

    ai, data, mask = load_integrator_and_data(poni_file, tif_file, mask_thresh)
    I2d, q, chi    = integrate_2d(ai, data, mask,
                                  num_azim_bins, q_min_nm1, npt_rad)
    # shift and reorder chi to [0,360)
    chi360 = (chi + 360) % 360
    order  = np.argsort(chi360)
    chi360 = chi360[order]

    q_peaks, pw, wq = detect_global_peaks(I2d, q,
                                          delta_tol=delta_tol, eta0=eta0)
    q_vs_chi       = fit_slices_parallel(I2d, q_peaks, pw, wq, q,
                                         delta_tol=delta_tol, eta0=eta0)
    q_vs_chi       = q_vs_chi[:, order]

    # # plot fitted centroids
    # plot_fitted_q_vs_chi(chi360, q_vs_chi, delta_tol, harmonics,
    #                      output_path=qazi_png)
    # # radar plot
    # plot_radar_all_rings(chi360, q_vs_chi, harmonics,
    #                      output_path=radar_png")

    # compute and print full strain tensor
    tensor = compute_strain_tensor(q_vs_chi, chi360, ai, harmonics)
    print("Computed strain tensor (ε_ij):")
    print(tensor)

if __name__ == "__main__":
    main()
