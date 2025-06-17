import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import pyFAI, fabio

# --- Pseudo-Voigt profile definition --------------------------------------
def pseudo_voigt(x, amp, cen, wid, eta):
    """
    Pseudo-Voigt profile:
      - amp: amplitude
      - cen: center (q) position
      - wid: full-width at half-maximum (FWHM)
      - eta: Lorentzian fraction (0: Gaussian, 1: Lorentzian)
    """
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))  # Gaussian sigma
    gamma = wid / 2                            # Lorentzian gamma
    gauss   = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

# --- 1) Load integrator and image ------------------------------------------
ai   = pyFAI.load("calibration/Calibration_LaB6_100x100_3s_r8.poni")
img  = fabio.open("InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif")
data = img.data
mask = data > 4e9

# --- 2) 2D integration with q >= 16 nm^-1 ---------------------------------
# quick radial-axis grab to get full q-array
numbins = 360
_q_full = ai.integrate2d(data, 1, 1, unit="q_nm^-1").radial
data_range = (16.0, _q_full[-1])
res2d = ai.integrate2d(
    data,
    npt_rad=5000,
    npt_azim=numbins,
    unit="q_nm^-1",
    mask=mask,
    radial_range=data_range,
)
I2d = res2d.intensity    # shape (npt_azim, npt_rad)
q   = res2d.radial       # length npt_rad
chi = res2d.azimuthal    # length npt_azim

# transpose if needed
if I2d.shape == (len(q), len(chi)):
    I2d = I2d.T

# --- 3) Global peak detection & pseudo-Voigt refinement -------------------
radial_mean = I2d.mean(axis=0)
# initial peak find
init_inds, _  = find_peaks(
    radial_mean,
    height=np.max(radial_mean)*0.09,
    distance=20
)
# estimate raw FWHM from half-max widths
widths_bins = peak_widths(radial_mean, init_inds, rel_height=0.5)[0]
q_initials  = q[init_inds]
# select the first 8 rings by ascending q
sel         = np.argsort(q_initials)[:8]
init_inds   = init_inds[sel]
q_initials  = q_initials[sel]
widths_q    = widths_bins[sel] * (q[1] - q[0])  # convert bin widths to q
eta0        = 0.5                                 # initial Lorentz fraction
delta_tol   = 0.05                                # noise tolerance in nm^-1

# refine global centers via pseudo-Voigt fits
q_peaks       = []
peak_windows  = []
for idx0, q0, wid0 in zip(init_inds, q_initials, widths_q):
    half_bins = int(np.ceil(wid0 / (q[1] - q[0])))
    wl = slice(max(0, idx0-half_bins), min(len(q), idx0+half_bins+1))
    x  = q[wl]
    y  = radial_mean[wl]
    try:
        p0   = [y.max(), q0, wid0, eta0]
        bounds = ([0, q0-delta_tol, 0, 0], [np.inf, q0+delta_tol, np.inf, 1])
        popt, _ = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds)
        qc = popt[1]
    except Exception:
        qc = q0
    q_peaks.append(qc)
    # find central bin of refined center
    i_cen = np.argmin(np.abs(q - qc))
    bw    = max(2, half_bins)
    peak_windows.append(slice(i_cen-bw, i_cen+bw+1))
q_peaks = np.array(q_peaks)

# --- 4) Slice-level pseudo-Voigt fitting with parallel execution -----------
def fit_slice(int_row):
    out = []
    for wl, q0, wid0 in zip(peak_windows, q_peaks, widths_q):
        x = q[wl]
        y = int_row[wl]
        if len(x) < 5:
            out.append(np.nan)
            continue
        p0 = [y.max(), q0, wid0, eta0]
        try:
            popt, _ = curve_fit(
                pseudo_voigt, x, y, p0=p0,
                bounds=([0, q0-delta_tol, 0, 0], [np.inf, q0+delta_tol, np.inf, 1]),
                maxfev=1000
            )
            out.append(popt[1])
        except Exception:
            out.append(np.nan)
    return out

# parallel map across chi-slices
q_vs_chi = Parallel(n_jobs=-1)(delayed(fit_slice)(row) for row in I2d)
q_vs_chi = np.array(q_vs_chi).T  # shape (n_peaks, n_slices)

# --- 5) Stacked-plot of Q vs azimuth --------------------------------------
fig, axes = plt.subplots(len(q_peaks), 1, sharex=True, figsize=(6, 2*len(q_peaks)))
for j, ax in enumerate(axes):
    ax.plot(chi, q_vs_chi[j], '-k', lw=1)
    m = np.nanmean(q_vs_chi[j])
    ax.set_ylim(m - delta_tol, m + delta_tol)
    ax.set_ylabel(f"q ≃ {q_peaks[j]:.2f}")
    ax.set_xlim(-180,180)
axes[-1].set_xlabel('Azimuth (°)')
fig.tight_layout()
plt.savefig(f"OutputFiles/QvsAz/SSAO-6_25C_Map_100x100_{numbins}bin.png")
plt.show()