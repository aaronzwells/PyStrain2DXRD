import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import pyFAI, fabio

# --- Pseudo‑Voigt profile definition --------------------------------------
def pseudo_voigt(x, amp, cen, wid, eta):
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    gamma = wid / 2
    gauss   = amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
    lorentz = amp * (gamma ** 2) / ((x - cen) ** 2 + gamma ** 2)
    return eta * lorentz + (1 - eta) * gauss

# --- User settings ----------------------------------------------------------
calib_poni   = "calibration/Calibration_LaB6_100x100_3s_r8.poni"
ref_tif      = "InputFiles/AVG_AO_Ref_100x100_AvgStack.tif"   # your stress‑free standard
sample_tif   = "InputFiles/VB-APS-SSAO-6_25C_Map-AO_000509.avg_BC.tif"
q_min        = 16.0      # nm⁻¹ lower limit
npt_rad      = 5000
npt_azim     = 120
eta0         = 0.5       # initial Lorentzian fraction
delta_tol    = 0.05      # fit window ±nm⁻¹

# --- 1) Load integrator -----------------------------------------------------
ai = pyFAI.load(calib_poni)

# determine full q-axis once
_q_full = ai.integrate2d(fabio.open(ref_tif).data, 1, 1, unit="q_nm^-1").radial
radial_range = (q_min, _q_full[-1])

# --- 2) Integrate reference pattern & find q0 --------------------------------
img_ref  = fabio.open(ref_tif)
data_ref = img_ref.data
mask_ref = data_ref > 4e9

res_ref = ai.integrate2d(
    data_ref,
    npt_rad=npt_rad,
    npt_azim=npt_azim,
    unit="q_nm^-1",
    mask=mask_ref,
    radial_range=radial_range,
)
I2d_ref = res_ref.intensity
q       = res_ref.radial
chi     = res_ref.azimuthal
if I2d_ref.shape == (len(q), len(chi)):
    I2d_ref = I2d_ref.T

# global mean over chi for ref
radial_mean_ref = I2d_ref.mean(axis=0)

# initial peak detection on reference
init_inds, _   = find_peaks(
    radial_mean_ref,
    height=np.max(radial_mean_ref)*0.09,
    distance=20
)
# get FWHM in q‑units
widths_ref   = peak_widths(radial_mean_ref, init_inds, rel_height=0.5)[0] * (q[1]-q[0])
q_initials   = q[init_inds]
# pick first 8 rings by ascending q:
sel          = np.argsort(q_initials)[:8]
init_inds    = init_inds[sel]
q_initials   = q_initials[sel]
widths_q     = widths_ref[sel]

# refine reference peak centers
q0_peaks     = []
peak_windows = []
for idx0, q0_guess, wid0 in zip(init_inds, q_initials, widths_q):
    half_bins = int(np.ceil(wid0/(q[1]-q[0])))
    wl = slice(max(0, idx0-half_bins), min(len(q), idx0+half_bins+1))
    x, y       = q[wl], radial_mean_ref[wl]
    try:
        p0     = [y.max(), q0_guess, wid0, eta0]
        bnds   = ([0, q0_guess-delta_tol, 0, 0], [np.inf, q0_guess+delta_tol, np.inf, 1])
        popt, _= curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bnds)
        qc     = popt[1]
    except:
        qc     = q0_guess
    q0_peaks.append(qc)
    icen = np.argmin(np.abs(q - qc))
    bw   = max(2, half_bins)
    peak_windows.append(slice(icen-bw, icen+bw+1))

q0_peaks = np.array(q0_peaks)   # shape (8,)

# --- 3) Integrate sample pattern & fit rings -------------------------------
img_samp  = fabio.open(sample_tif)
data_samp = img_samp.data
mask_samp = data_samp > 4e9

res_samp = ai.integrate2d(
    data_samp,
    npt_rad=npt_rad,
    npt_azim=npt_azim,
    unit="q_nm^-1",
    mask=mask_samp,
    radial_range=radial_range,
)
I2d_samp = res_samp.intensity
if I2d_samp.shape == (len(q), len(chi)):
    I2d_samp = I2d_samp.T

def fit_slice(int_row):
    out = []
    for wl, q0, wid0 in zip(peak_windows, q0_peaks, widths_q):
        x, y = q[wl], int_row[wl]
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
        except:
            out.append(np.nan)
    return out

# parallel fit across azimuth slices
q_vs_chi_samp = Parallel(n_jobs=-1)(delayed(fit_slice)(row) for row in I2d_samp)
q_vs_chi_samp = np.array(q_vs_chi_samp).T   # (n_peaks, n_slices)

# --- 4) Compute strain & plot ---------------------------------------------
# strain ε = –Δq/q0
strain_vs_chi = - (q_vs_chi_samp - q0_peaks[:, None]) / q0_peaks[:, None]

fig, axes = plt.subplots(len(q0_peaks), 1, sharex=True, figsize=(6, 2*len(q0_peaks)))
for j, ax in enumerate(axes):
    ax.plot(chi, strain_vs_chi[j], '-k', lw=1)
    mean_eps = np.nanmean(strain_vs_chi[j])
    eps_tol  = delta_tol / q0_peaks[j]
    ax.set_ylim(mean_eps - eps_tol, mean_eps + eps_tol)
    ax.set_ylabel(f"ε ≃ {mean_eps:.2e}")
axes[-1].set_xlabel('Azimuth (°)')
fig.tight_layout()
plt.savefig("OutputFiles/Sample_StrainVsAzPlots.png")
plt.show()
