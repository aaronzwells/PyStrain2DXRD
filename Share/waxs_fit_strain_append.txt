#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Stage 3: fit the He & Smith (1998) lattice-cone distortion model to the peak
centers in an existing fit-results HDF5 file, and APPEND the strain results as
a new top-level group.  Nothing under /fit_results is ever touched.

Reads   /fit_results/eta
        /fit_results/<phase>/{params, errors, mask, chi2}   [frame, peak, eta, ...]
        /fit_results/metadata (attrs: radial_unit, radial_scale)

Writes  /strain_fit/<phase>/<N>comp/
            q_fit        (n_frames, n_peaks, n_eta)  fitted q(eta), file radial units
            q0_ref       (n_peaks, n_eta)            reference q0(eta), file radial units
            eta          (n_eta,)
            used_mask    (n_frames, n_peaks, n_eta)  bins kept after outlier rejection
            strain       (n_frames, n_peaks, 6)      eps_xx, eps_xy, eps_yy, eps_xz, eps_yz, eps_zz
            strain_err   (n_frames, n_peaks, 6)
            red_chi2     (n_frames, n_peaks)
            rms_resid    (n_frames, n_peaks)         unweighted rms of ln(q0/q) residuals
            cond         (n_frames, n_peaks)         condition number of the weighted design matrix
            n_used       (n_frames, n_peaks)

With --joint, all peaks of a frame (or the subset given by --peaks) are fitted
together with ONE shared strain tensor per frame, and the group is named
/strain_fit/<phase>/<N>comp_joint/ with these shape changes:
            strain, strain_err   (n_frames, 6)
            red_chi2, cond, n_used (n_frames,)
            rms_resid, ring_offset (n_frames, n_peaks)  per-peak residual rms / median offset
q_fit, q0_ref, used_mask, eta keep the per-peak shapes above, so the viewer
overlay works unchanged.

Components that are not part of the chosen model (e.g. eps_xz for 3comp) are NaN.
Strain convention: eps = ln(q0/q) = ln(d/d0), so positive = lattice expansion.

Example Usage:
    python waxs_fit_strain_append.py scan_fitresults.h5 --wavelength-nm 0.0123 \
        --ref-frame 0 --num-comp 3 --phase Fe
"""
import argparse
import time

import h5py
import numpy as np

# Index of the peak-center parameter in params[..., k]. errors[..., k] is its uncertainty.
# CHECK THIS against PARAM_LABELS in the viewer.
CENTER_IDX = 1

COMPONENT_NAMES = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']
COMPONENT_COLS = {3: [0, 1, 2], 5: [0, 1, 2, 3, 4], 6: [0, 1, 2, 3, 4, 5]}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def design_matrix(eta_deg, q0_nm, wavelength_nm, psi_deg=0.0, phi_deg=0.0, omega_deg=90.0):
    """Columns f11, f12, f22, f13, f23, f33 for each eta bin (He & Smith 1998).

    Theta is computed from the reference q0 so that the fit and the predicted
    curve use exactly the same matrix.  q0 must be in nm^-1, wavelength in nm.
    """
    chi = np.deg2rad(90.0 - np.asarray(eta_deg, float))  # same transform as the original function (A. Wells)
    psi, phi, om = np.deg2rad([psi_deg, phi_deg, omega_deg])
    with np.errstate(invalid='ignore'):
        theta = np.arcsin(np.asarray(q0_nm, float) * wavelength_nm / (4 * np.pi))
    st, ct = np.sin(theta), np.cos(theta)
    sc, cc = np.sin(chi), np.cos(chi)

    a = st * np.cos(om) + sc * ct * np.sin(om)
    b = -cc * ct
    c = st * np.sin(om) - sc * ct * np.cos(om)

    A = a * np.cos(phi) - b * np.cos(psi) * np.sin(phi) + c * np.sin(psi) * np.sin(phi)
    B = a * np.sin(phi) + b * np.cos(psi) * np.cos(phi) - c * np.sin(psi) * np.cos(phi)
    C = b * np.sin(psi) + c * np.cos(psi)
    return np.column_stack([A * A, 2 * A * B, B * B, 2 * A * C, 2 * B * C, C * C])


def _wls(F, y, sig):
    """Weighted least squares. Errors are scaled by the residual variance (same as statsmodels bse)."""
    w = 1.0 / sig
    Fw, yw = F * w[:, None], y * w
    coef, *_ = np.linalg.lstsq(Fw, yw, rcond=None)
    dof = len(y) - F.shape[1]
    r = yw - Fw @ coef
    red_chi2 = (r @ r) / dof
    cov = np.linalg.pinv(Fw.T @ Fw) * red_chi2
    return coef, np.sqrt(np.diag(cov)), red_chi2, np.linalg.cond(Fw)


def fit_ring(eta_deg, q_nm, qerr_nm, q0_nm, cols, wavelength_nm, angles,
             mad_threshold=5.0, min_points=20, max_iter=3):
    """Fit one (frame, peak) q(eta) curve. Returns a dict, or None if it cannot be fitted.

    Outlier rejection works on the RESIDUALS of the model (iterated), using a
    robust sigma = 1.4826 * MAD.  mad_threshold is therefore in units of sigma
    (8 raw MADs in the old function is about 5.4 sigma).
    """
    Fall = design_matrix(eta_deg, q0_nm, wavelength_nm, *angles)
    F = Fall[:, cols]

    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.log(q0_nm / q_nm)
        sig = np.maximum(qerr_nm / q_nm, 1e-6)
    ok = np.isfinite(y) & np.isfinite(sig) & np.isfinite(F).all(axis=1)
    if ok.sum() < max(min_points, len(cols) + 1):
        return None

    use = ok.copy()
    for it in range(max_iter + 1):
        if use.sum() < max(min_points, len(cols) + 1):
            return None
        coef, err, red_chi2, cond = _wls(F[use], y[use], sig[use])
        if it == max_iter:
            break
        r = y - F @ coef
        med = np.median(r[use])
        s = 1.4826 * np.median(np.abs(r[use] - med))
        if not s > 0:
            break
        new = ok & (np.abs(r - med) < mad_threshold * s)
        if np.array_equal(new, use):
            break
        use = new

    with np.errstate(invalid='ignore'):
        q_fit = q0_nm / np.exp(Fall[:, cols] @ coef)
    resid = (y - F @ coef)[use]
    return dict(coef=coef, err=err, red_chi2=red_chi2, cond=cond, use=use,
                q_fit=q_fit, rms=float(np.sqrt(np.mean(resid ** 2))))


def fit_frame_joint(eta_deg, q_nm, qerr_nm, q0_nm, cols, wavelength_nm, angles,
                    mad_threshold=5.0, min_points=20, max_iter=3):
    """Fit ONE strain tensor to all peaks of a frame at once.

    q_nm, qerr_nm, q0_nm have shape (n_peaks, n_eta).  Each peak keeps its own
    design matrix (its own theta, from its own q0), but all rows are stacked and
    share the same strain components.  Outlier rejection is done per peak, so a
    noisy peak does not set the threshold for a clean one.
    """
    n_peaks, n_eta = q_nm.shape
    npar = len(cols)
    Fall = np.stack([design_matrix(eta_deg, q0_nm[p], wavelength_nm, *angles)
                     for p in range(n_peaks)])            # (n_peaks, n_eta, 6)
    F = Fall[:, :, cols]

    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.log(q0_nm / q_nm)
        sig = np.maximum(qerr_nm / q_nm, 1e-6)
    ok = np.isfinite(y) & np.isfinite(sig) & np.isfinite(F).all(axis=2)
    if ok.sum() < max(min_points, npar + 1):
        return None

    use = ok.copy()
    for it in range(max_iter + 1):
        if use.sum() < max(min_points, npar + 1):
            return None
        coef, err, red_chi2, cond = _wls(F[use], y[use], sig[use])   # boolean mask stacks all peaks
        if it == max_iter:
            break
        r = y - F @ coef                                              # (n_peaks, n_eta)
        new = np.zeros_like(use)
        for p in range(n_peaks):
            if use[p].sum() < 5:
                continue
            med = np.median(r[p, use[p]])                             # per-peak offset is not an outlier
            s = 1.4826 * np.median(np.abs(r[p, use[p]] - med))
            if s > 0:
                new[p] = ok[p] & (np.abs(r[p] - med) < mad_threshold * s)
            else:
                new[p] = use[p]
        if np.array_equal(new, use):
            break
        use = new

    with np.errstate(invalid='ignore'):
        q_fit = q0_nm / np.exp(F @ coef)
    r = y - F @ coef
    rms = np.full(n_peaks, np.nan)
    off = np.full(n_peaks, np.nan)
    for p in range(n_peaks):
        if use[p].any():
            rms[p] = np.sqrt(np.mean(r[p, use[p]] ** 2))
            off[p] = np.median(r[p, use[p]])
    return dict(coef=coef, err=err, red_chi2=red_chi2, cond=cond, use=use,
                q_fit=q_fit, rms=rms, offset=off)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def read_inputs(h5path, phase=None):
    """Read everything into memory, then close the file."""
    with h5py.File(h5path, 'r') as f:
        fr = f['fit_results']
        eta = fr['eta'][:]
        meta = {k: fr['metadata'].attrs[k] for k in ('radial_unit', 'radial_scale')
                if k in fr['metadata'].attrs}
        if phase:
            phases = [phase]
        else:
            phases = [k for k in fr.keys() if k not in ('eta', 'metadata')]
        data = {}
        for p in phases:
            g = fr[p]
            data[p] = dict(params=g['params'][:], errors=g['errors'][:],
                           mask=g['mask'][:].astype(bool), chi2=g['chi2'][:])
    return eta, meta, data


def _prepare(d, ref_frame, q_factor, max_chi2):
    """Apply masks, convert to nm^-1 and pull out the reference-frame q0(eta)."""
    params, errors, mask, chi2 = d['params'], d['errors'], d['mask'], d['chi2']
    n_frames = params.shape[0]

    valid = mask.copy()
    if max_chi2 is not None:
        with np.errstate(invalid='ignore'):
            valid &= chi2 <= max_chi2

    q_all = params[..., CENTER_IDX] * q_factor
    e_all = errors[..., CENTER_IDX] * q_factor
    q_all = np.where(valid, q_all, np.nan)

    if not 0 <= ref_frame < n_frames:
        raise ValueError(f"ref_frame {ref_frame} out of range (0..{n_frames - 1})")
    q_ref = q_all[ref_frame]                                  # (n_peaks, n_eta), nm^-1
    if not np.isfinite(q_ref).any():
        raise ValueError(f"reference frame {ref_frame} has no valid bins")
    return q_all, e_all, q_ref


def fit_phase(eta, d, ref_frame, num_comp, wavelength_nm, q_factor, angles,
              mad_threshold, min_points, max_chi2):
    """Run the fit for every (frame, peak) of one phase. q_factor converts file units to nm^-1."""
    q_all, e_all, q_ref = _prepare(d, ref_frame, q_factor, max_chi2)
    n_frames, n_peaks, n_eta = q_all.shape
    cols = COMPONENT_COLS[num_comp]

    out = dict(
        q_fit=np.full((n_frames, n_peaks, n_eta), np.nan),
        used_mask=np.zeros((n_frames, n_peaks, n_eta), bool),
        strain=np.full((n_frames, n_peaks, 6), np.nan),
        strain_err=np.full((n_frames, n_peaks, 6), np.nan),
        red_chi2=np.full((n_frames, n_peaks), np.nan),
        rms_resid=np.full((n_frames, n_peaks), np.nan),
        cond=np.full((n_frames, n_peaks), np.nan),
        n_used=np.zeros((n_frames, n_peaks), int),
    )
    n_fail = 0
    for fi in range(n_frames):
        for pi in range(n_peaks):
            res = fit_ring(eta, q_all[fi, pi], e_all[fi, pi], q_ref[pi], cols,
                           wavelength_nm, angles, mad_threshold, min_points)
            if res is None:
                n_fail += 1
                continue
            out['q_fit'][fi, pi] = res['q_fit']
            out['used_mask'][fi, pi] = res['use']
            out['strain'][fi, pi, cols] = res['coef']
            out['strain_err'][fi, pi, cols] = res['err']
            out['red_chi2'][fi, pi] = res['red_chi2']
            out['rms_resid'][fi, pi] = res['rms']
            out['cond'][fi, pi] = res['cond']
            out['n_used'][fi, pi] = res['use'].sum()
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"    frame {fi + 1}/{n_frames}")

    # store q in the FILE's radial units so the viewer's ln(q_fit / ref) is consistent
    out['q_fit'] /= q_factor
    out['q0_ref'] = q_ref / q_factor
    out['n_fail'] = n_fail
    out['n_total'] = n_frames * n_peaks
    return out


def fit_phase_joint(eta, d, ref_frame, num_comp, wavelength_nm, q_factor, angles,
                    mad_threshold, min_points, max_chi2, peaks=None):
    """One shared strain tensor per frame, fitted to all peaks (or the subset `peaks`) at once."""
    q_all, e_all, q_ref = _prepare(d, ref_frame, q_factor, max_chi2)
    n_frames, n_peaks, n_eta = q_all.shape
    cols = COMPONENT_COLS[num_comp]
    sel = np.arange(n_peaks) if peaks is None else np.asarray(peaks, int)
    if sel.min() < 0 or sel.max() >= n_peaks:
        raise ValueError(f"peaks must be in 0..{n_peaks - 1}")

    out = dict(
        q_fit=np.full((n_frames, n_peaks, n_eta), np.nan),
        used_mask=np.zeros((n_frames, n_peaks, n_eta), bool),
        strain=np.full((n_frames, 6), np.nan),
        strain_err=np.full((n_frames, 6), np.nan),
        red_chi2=np.full(n_frames, np.nan),
        cond=np.full(n_frames, np.nan),
        n_used=np.zeros(n_frames, int),
        rms_resid=np.full((n_frames, n_peaks), np.nan),
        ring_offset=np.full((n_frames, n_peaks), np.nan),
    )
    n_fail = 0
    for fi in range(n_frames):
        res = fit_frame_joint(eta, q_all[fi, sel], e_all[fi, sel], q_ref[sel], cols,
                              wavelength_nm, angles, mad_threshold, min_points)
        if res is None:
            n_fail += 1
            continue
        out['q_fit'][fi, sel] = res['q_fit']
        out['used_mask'][fi, sel] = res['use']
        out['strain'][fi, cols] = res['coef']
        out['strain_err'][fi, cols] = res['err']
        out['red_chi2'][fi] = res['red_chi2']
        out['cond'][fi] = res['cond']
        out['n_used'][fi] = res['use'].sum()
        out['rms_resid'][fi, sel] = res['rms']
        out['ring_offset'][fi, sel] = res['offset']
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"    frame {fi + 1}/{n_frames}")

    out['q_fit'] /= q_factor                                  # back to the file's radial units
    q_ref_store = np.full_like(q_ref, np.nan)
    q_ref_store[sel] = q_ref[sel]
    out['q0_ref'] = q_ref_store / q_factor
    out['n_fail'] = n_fail
    out['n_total'] = n_frames
    return out


def write_results(h5path, phase, name, eta, out, attrs):
    """Append /strain_fit/<phase>/<name>; replaces only that one group."""
    with h5py.File(h5path, 'a') as f:
        pg = f.require_group('strain_fit').require_group(phase)
        if name in pg:
            del pg[name]
        g = pg.create_group(name)
        for k, v in attrs.items():
            g.attrs[k] = v
        g.attrs['component_names'] = ','.join(COMPONENT_NAMES)
        g.create_dataset('eta', data=eta)
        g.create_dataset('q0_ref', data=out['q0_ref'])
        for key in ('q_fit', 'used_mask', 'strain', 'strain_err',
                    'red_chi2', 'rms_resid', 'ring_offset', 'cond', 'n_used'):
            if key in out:
                g.create_dataset(key, data=out[key], compression='gzip')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(h5path, wavelength_nm, phase=None, ref_frame=0, num_comp=3, q_factor=1.0,
         psi_deg=0.0, phi_deg=0.0, omega_deg=90.0, mad_threshold=5.0,
         min_points=20, max_chi2=None, joint=False, peaks=None):
    if num_comp not in COMPONENT_COLS:
        raise ValueError("num_comp must be 3, 5 or 6")

    eta, meta, data = read_inputs(h5path, phase)
    print(f"File radial_unit={meta.get('radial_unit')!r}, radial_scale={meta.get('radial_scale')!r}; "
          f"using q_factor={q_factor} to convert to nm^-1")
    angles = (psi_deg, phi_deg, omega_deg)

    for p, d in data.items():
        print(f"\nPhase {p}: params {d['params'].shape}, ref frame {ref_frame}, {num_comp} components")
        t0 = time.time()
        if joint:
            out = fit_phase_joint(eta, d, ref_frame, num_comp, wavelength_nm, q_factor, angles,
                                  mad_threshold, min_points, max_chi2, peaks)
            unit_name, name = 'frames (all peaks together)', f'{num_comp}comp_joint'
            if peaks is not None:   # keep subset runs from overwriting the all-peaks group
                name += '_p' + '_'.join(str(int(k)) for k in peaks)
        else:
            out = fit_phase(eta, d, ref_frame, num_comp, wavelength_nm, q_factor, angles,
                            mad_threshold, min_points, max_chi2)
            unit_name, name = '(frame, peak) curves', f'{num_comp}comp'
        print(f"  fitted {out['n_total'] - out['n_fail']}/{out['n_total']} {unit_name} "
              f"in {time.time() - t0:.1f}s")
        if num_comp == 6:
            print(f"  median condition number: {np.nanmedian(out['cond']):.3g} "
                  f"(very large => eps_zz is not separable with this geometry)")
        attrs = dict(num_strain_components=num_comp, ref_frame=ref_frame,
                     wavelength_nm=wavelength_nm, q_factor_to_nm_inv=q_factor,
                     psi_deg=psi_deg, phi_deg=phi_deg, omega_deg=omega_deg,
                     mad_threshold_sigma=mad_threshold, min_points=min_points,
                     max_chi2=-1.0 if max_chi2 is None else max_chi2,
                     joint=bool(joint),
                     peaks=-1 if peaks is None else np.asarray(peaks, int),
                     center_idx=CENTER_IDX,
                     radial_unit=str(meta.get('radial_unit', '')),
                     radial_scale=float(meta.get('radial_scale', 1.0)),
                     created=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_results(h5path, p, name, eta, out, attrs)
        print(f"  wrote /strain_fit/{p}/{name}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('h5file')
    ap.add_argument('--wavelength-nm', type=float, required=True)
    ap.add_argument('--phase', default=None, help='default: all phases in the file')
    ap.add_argument('--ref-frame', type=int, default=0)
    ap.add_argument('--num-comp', type=int, default=3, choices=[3, 5, 6])
    ap.add_argument('--q-factor', type=float, default=1.0,
                    help='multiply file radial values by this to get nm^-1 (e.g. 10 if the file is in A^-1)')
    ap.add_argument('--psi-deg', type=float, default=0.0)
    ap.add_argument('--phi-deg', type=float, default=0.0)
    ap.add_argument('--omega-deg', type=float, default=90.0)
    ap.add_argument('--mad-threshold', type=float, default=5.0, help='in robust sigma units')
    ap.add_argument('--min-points', type=int, default=20)
    ap.add_argument('--max-chi2', type=float, default=None, help='drop bins with peak-fit chi2 above this')
    ap.add_argument('--joint', action='store_true',
                    help='fit all peaks together with one shared strain tensor per frame')
    ap.add_argument('--peaks', type=int, nargs='+', default=None,
                    help='with --joint: peak indices to include (default: all)')
    a = ap.parse_args()
    main(a.h5file, a.wavelength_nm, a.phase, a.ref_frame, a.num_comp, a.q_factor,
         a.psi_deg, a.phi_deg, a.omega_deg, a.mad_threshold, a.min_points, a.max_chi2,
         a.joint, a.peaks)







# """
# Fits lattice cone distortion in a script and appends to the peak fit hdf5.
# Will modify the viewer later to display the outputs
# """

# import argparse
# import os
# import sys
# import time
# import numpy as np
# import h5py

# #Directory Parameters
# visit = "Feb2025"
# isolated_mapscan_location = "Feb2025_OnHeat_25C" #Grouping the maps for organization
# h5_file = "scan_range_169_520_fit_max10.h5" #Name of the hdf5 file to append the fit results to
# scan_range = (169, 520)

# h5path = os.path.join("2_BinnedIntegrationAndFitting", visit, isolated_mapscan_location, h5_file)


# def main(h5path, phase, ref_frame, num_comp=3, wl_nm=..., group_name=None):
#     group_name = group_name or f'{phase}_strain_fit_{num_comp}comp'

#     # 1. READ (open read-only, then close before writing)
#     with h5py.File(h5path, 'r') as f:
#         # load into memory: params, errors, mask, eta_axis for this phase
#         # (use whatever paths/datasets your results file uses)
#         ...

#     # 2. FIT, one call per frame
#     q_ref = params[ref_frame, :, :, 1].copy()
#     q_ref[~mask[ref_frame]] = np.nan
#     n_frames, n_peaks, n_eta = ...
#     q_fit    = np.full((n_frames, n_peaks, n_eta), np.nan)
#     strain   = np.full((n_frames, n_peaks, 6), np.nan)
#     strain_e = np.full((n_frames, n_peaks, 6), np.nan)
#     for fi in range(n_frames):
#         q = params[fi, :, :, 1].copy();  q[~mask[fi]] = np.nan
#         e = errors[fi, :, :, 1].copy()
#         ...  # call the fit, fill q_fit[fi], strain[fi], strain_e[fi]

#     # 3. WRITE (append mode, replace only your own group)
#     with h5py.File(h5path, 'a') as f:
#         if group_name in f: del f[group_name]
#         g = f.create_group(group_name)
#         g.attrs.update(num_strain_components=num_comp, ref_frame=ref_frame,
#                        wavelength_nm=wl_nm)
#         g.create_dataset('q_fit', data=q_fit, compression='gzip')
#         g.create_dataset('q0_ref', data=q_ref)
#         g.create_dataset('eta_deg', data=eta_axis)
#         g.create_dataset('strain', data=strain)
#         g.create_dataset('strain_err', data=strain_e)



# #############################################################################
# # This block modified from A. Wells "fit_lattice_cone_distortion"
# #############################################################################

# def fit_lattice_cone_distortion(q_data, q_errors, q0_chi_data, initial_q_guesses, wavelength_nm,
#                                 chi_deg=None, psi_deg=None, phi_deg=None, omega_deg=None, num_strain_components=3, MAD_threshold=8.0,
#                                 output_dir=None, dpi=600, plot=True, logger=None, min_rsquared=0.0):
#     """
#     Fits a lattice cone distortion model to q(χ) data to extract strain tensor components.

#     This function implements the model described by He & Smith (1998) to determine
#     the components of the strain tensor (ε_ij) by fitting the variation of the
#     diffraction ring radius (q) with azimuthal angle (χ).

#     Args:
#         q_data (np.ndarray): Array of q centroids of shape (n_rings, n_chi_bins).
#         q_errors (np.ndarray): Array of q centroid errors of shape (n_rings, n_chi_bins).
#         q0_chi_data (np.ndarray): Array of unstrained lattice spacings (q0) for each ring.
#         initial_q_guesses (list): List of initial q values, used for plot titles.
#         wavelength_nm (float): X-ray wavelength in nanometers.
#         chi_deg (np.ndarray, optional): Azimuthal angles in degrees. Defaults to a uniform grid.
#         psi_deg, phi_deg, omega_deg (np.ndarray, optional): Sample orientation angles. Defaults to a simple geometry.
#         num_strain_components (int, optional): Number of strain components to solve for (3, 5, or 6). Defaults to 3.
#         output_dir (str, optional): Directory to save plots and results. Defaults to None.
#         dpi (int, optional): Resolution for saved plots. Defaults to 600.
#         plot (bool, optional): If True, generate and save plots. Defaults to True.
#         logger (logging.Logger, optional): Logger for status messages. Defaults to None.
#         min_rsquared (float, optional): Minimum R-squared value to accept a fit. Defaults to 0.5.
#     Returns:
#         tuple: (strain_params, strain_list, q0_list_out, strain_vs_chi_path)
#     """
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm
#     logger = logger or logging.getLogger(__name__)
    
#     plt.rcParams.update({'font.size': 16})

#     os.makedirs(output_dir, exist_ok=True)
#     # collect fitted strain-vs-chi curves for overlay
#     fit_vs_chi = []
#     q_fit_vs_chi_list = []

#     # q_data is provided as a numpy array of shape (n_rings, n_bins)
#     n_rings, n_bins = q_data.shape

#     if chi_deg is None:
#         chi_deg = np.linspace(0, 360, n_bins, endpoint=False)
#     # set default orientation angles if not provided
#     if psi_deg is None:
#         psi_deg_full = np.full_like(chi_deg, 0.0) # ψ = 0°
#     if phi_deg is None:
#         phi_deg_full = np.full_like(chi_deg, 0.0) # φ = 0°
#     if omega_deg is None:
#         omega_deg_full = np.full_like(chi_deg, 90.0) # ω = 90°

#     strain_params = []
#     fig, axes = (plt.subplots(n_rings, 1, figsize=(10, 2 * n_rings), dpi=dpi, sharex=True) if plot else (None, None))
    
#     if plot and n_rings == 1: axes = [axes]

#     # y limits for the strain vs chi plots
#     y_min, y_max = -0.0015, 0.0015

#     nan_dict = {key: np.nan for key in ['q0', 'eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz', 
#                                         'eps_xx_err', 'eps_xy_err', 'eps_yy_err', 'eps_xz_err', 'eps_yz_err', 'eps_zz_err']}

#     for i in range(n_rings):
#         q_vals = q_data[i]
#         q_errs = q_errors[i]
#         q0_vals_full = q0_chi_data[i] 
        
#         # Defining q0_fixed for naming of plots
#         q0_fixed = initial_q_guesses[i]

#         # --- Calculate the FULL-RANGE geometric design matrix (F_all_full) ---
#         # This is used later to predict the smooth fitted curve
#         try:
#             chi_rad_full   = np.deg2rad(90-chi_deg)
#             psi_rad_full   = np.deg2rad(psi_deg_full)
#             phi_rad_full   = np.deg2rad(phi_deg_full)
#             omega_rad_full = np.deg2rad(omega_deg_full)
            
#             # Use the FULL q0(chi) reference to calculate theta for the model
#             theta_full     = np.arcsin((q0_vals_full * wavelength_nm) / (4 * np.pi))
#             sin_chi_full   = np.sin(chi_rad_full)
#             cos_chi_full   = np.cos(chi_rad_full)
#             sin_theta_full = np.sin(theta_full)
#             cos_theta_full = np.cos(theta_full)

#             a_full = sin_theta_full * np.cos(omega_rad_full) + sin_chi_full * cos_theta_full * np.sin(omega_rad_full)
#             b_full = -cos_chi_full * cos_theta_full
#             c_full = sin_theta_full * np.sin(omega_rad_full) - sin_chi_full * cos_theta_full * np.cos(omega_rad_full)
            
#             A_full = a_full * np.cos(phi_rad_full) - b_full * np.cos(psi_rad_full) * np.sin(phi_rad_full) + c_full * np.sin(psi_rad_full) * np.sin(phi_rad_full)
#             B_full = a_full * np.sin(phi_rad_full) + b_full * np.cos(psi_rad_full) * np.cos(phi_rad_full) - c_full * np.sin(psi_rad_full) * np.cos(phi_rad_full)
#             C_full = b_full * np.sin(psi_rad_full) + c_full * np.cos(psi_rad_full)
            
#             f11_full, f12_full, f22_full = A_full**2, 2*A_full*B_full, B_full**2
#             f13_full, f23_full, f33_full = 2*A_full*C_full, 2*B_full*C_full, C_full**2
            
#             F_all_full = np.vstack([f11_full, f12_full, f22_full, f13_full, f23_full, f33_full]).T
        
#         except Exception as e:
#             logger.error(f"Failed to build geometric model for Ring {i+1}. Skipping. Error: {e}")
#             strain_params.append(nan_dict.copy())
#             q_fit_vs_chi_list.append(np.full(n_bins, np.nan))
#             if plot:
#                 axes[i].set_title(f"Ring {i+1}: Model build failed"); axes[i].axis('off')
#             continue

#         # Filter out any NaN values from the input data
#         mask = ~np.isnan(q_vals) & ~np.isnan(q_errs) & ~np.isnan(q0_vals_full)
#         x, y, y_err, q0_masked = chi_deg[mask], q_vals[mask], q_errs[mask], q0_vals_full[mask]
        
#         # Filter outliers using the Median Absolute Deviation (MAD) method
#         if len(y) > 0: # Ensure there is data to filter
#             median_q = np.median(y)
#             abs_deviation = np.abs(y - median_q)
#             mad = np.median(abs_deviation)
#             threshold = MAD_threshold * mad # Define the outlier threshold (3.0 is a good starting point)
#             outlier_mask = abs_deviation < threshold # Keep only the points within the threshold
#             num_outliers = len(y) - np.sum(outlier_mask) # Log how many points were removed
#             if num_outliers > 0:
#                 logger.info(f"Ring {i+1}: Removed {num_outliers} outliers using MAD filter.")
                
#             # Apply outlier_mask to all data arrays, including q0_masked
#             x, y, y_err, q0_masked = x[outlier_mask], y[outlier_mask], y_err[outlier_mask], q0_masked[outlier_mask]
        
#         # Ensure a minimum error value to avoid division by zero in weights
#         min_error = 1e-6
#         y_err = np.maximum(y_err, min_error)

#         # Skip ring if there are not enough data points for a reliable fit
#         if len(x) < 20:
#             logger.warning(f"Ring {i+1}: insufficient data points ({len(x)} < 20). Skipping.")
#             if plot:
#                 axes[i].set_title(f"Ring {i+1}: insufficient data"); axes[i].axis('off')
#             strain_params.append(nan_dict.copy()) # creates a dictionary of NaN values if the ring is skipped
#             fit_vs_chi.append(np.full(n_bins, np.nan))
#             continue

#         try:
#             # --- Geometric Transformations and Model Setup ---
#             chi_rad   = np.deg2rad(90-x) # transform so χ is aligned with the coordinate system in He & Smith 1998
#             psi_rad   = np.deg2rad(psi_deg_full[mask][outlier_mask]) # Use consistent geometry
#             phi_rad   = np.deg2rad(phi_deg_full[mask][outlier_mask])
#             omega_rad = np.deg2rad(omega_deg_full[mask][outlier_mask])
#             # Compute θ for each centroid: θ = arcsin(q*λ/(4π))
#             theta     = np.arcsin((y * wavelength_nm) / (4 * np.pi))
#             sin_chi   = np.sin(chi_rad)
#             cos_chi   = np.cos(chi_rad)
#             sin_theta = np.sin(theta)
#             cos_theta = np.cos(theta)
#             # Intermediate parameters a, b, c from Table 1 He & Smith 1998
#             a = sin_theta * np.cos(omega_rad) + sin_chi * cos_theta * np.sin(omega_rad)
#             b = -cos_chi * cos_theta
#             c = sin_theta * np.sin(omega_rad) - sin_chi * cos_theta * np.cos(omega_rad)
#             # Direction cosines A, B, C
#             A = a * np.cos(phi_rad) - b * np.cos(psi_rad) * np.sin(phi_rad) + c * np.sin(psi_rad) * np.sin(phi_rad)
#             B = a * np.sin(phi_rad) + b * np.cos(psi_rad) * np.cos(phi_rad) - c * np.sin(psi_rad) * np.cos(phi_rad)
#             C = b * np.sin(psi_rad) + c * np.cos(psi_rad)
#             # Defining the strain coefficients: f_ij
#             f11, f12, f22, f13, f23, f33 = A**2, 2*A*B, B**2, 2*A*C, 2*B*C, C**2
            
#             # The model is y_meas = F * [strain_components]
#             F_all_masked = np.vstack([f11, f12, f22, f13, f23, f33]).T
#             y_meas = np.log(q0_masked / y)
            
#             # Propagate errors to get weights for the Weighted Least Squares (WLS) fit
#             # Error in y_meas = ln(q0/q) is approx. sigma_q / q
#             y_meas_err = y_err / y
#             y_meas_err[y_meas_err == 0] = min_error # Avoid division by zero
#             weights = 1.0 / (y_meas_err**2)
            
#             epsilon = '\u03B5' # ε
#             logger.info(f"The model is solving for {num_strain_components} strain components.")
#             # Select the appropriate columns of the design matrix F based on the desired model
#             if num_strain_components == 6:
#                 F_fit = F_all_masked
#                 F_model = F_all_full # Full model for prediction
#                 param_names = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']
#                 if i==1: logger.info(f"No strain components are set to 0")
#             elif num_strain_components == 5:
#                 F_fit = F_all_masked[:, [0, 1, 2, 3, 4]]
#                 F_model = F_all_full[:, [0, 1, 2, 3, 4]]
#                 param_names = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz']
#                 if i==1: logger.info(f"{epsilon}33 is set to zero")
#             elif num_strain_components == 3:
#                 F_fit = F_all_masked[:, [0, 1, 2]]
#                 F_model = F_all_full[:, [0, 1, 2]]
#                 param_names = ['eps_xx', 'eps_xy', 'eps_yy']
#                 if i==1: logger.info(f"{epsilon}13, {epsilon}23, & {epsilon}33 are set to zero")
#             else: 
#                 F = F_all[:, [0, 1, 2]]
#                 param_names = ['eps_xx', 'eps_xy', 'eps_yy']
#                 if i==1: logger.warning(f"An incompatible number of strain components was selected. The only valid numbers are 3 (biaxial), 5 (biaxial w/ shear) & 6 (full strain tensor). Defaulting to biaxial.")

#             # --- Perform WLS fit ---
#             wls_model = sm.WLS(y_meas, F_fit, weights=weights)
#             results = wls_model.fit()
            
#             # Filter out poor fits based on the R-squared value
#             if results.rsquared < min_rsquared:
#                 raise ValueError(f"R-squared ({results.rsquared:.3f}) is below the threshold of {min_rsquared}.")

#             # Store results and errors
#             params = {f'{p}': val for p, val in zip(param_names, results.params)}
#             errors = {f'{p}_err': err for p, err in zip(param_names, results.bse)}
            
#             # Fill in NaNs for components not in the fit
#             full_params = nan_dict.copy()
#             full_params['q0'] = np.mean(q0_masked)
#             full_params.update(params)
#             full_params.update(errors)
#             strain_params.append(full_params)

#             # --- Generate and save fitted q(chi) curve ---
#             fitted_params = results.params
#             eps_fit_full = F_model @ fitted_params
#             q_fit_full = q0_vals_full / np.exp(eps_fit_full)
#             q_fit_vs_chi_list.append(q_fit_full)

#             if plot:
#                 ax = axes[i]
#                 ax.plot(x, y_meas*1e6, '.', markersize=3, label='ln(q₀/q)')
#                 ax.plot(x, results.fittedvalues*1e6, '-', label='Fit') # Use results.fittedvalues
#                 ax.set_ylabel('Microstrain')
#                 # ax.set_ylabel(f'{epsilon}=ln(q₀/q)')
#                 ax.set_ylim(-1000,1000)
#                 ax.set_xlim(0, 360)
#                 # ax.set_title(f'Ring {i+1} (q₀ = {q0_fixed:.4f} nm⁻¹)')
#                 # ax.set_title(f'q₀ = {q0_fixed:.4f} nm⁻¹')
#                 ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0.02))
        
#         except (ValueError, np.linalg.LinAlgError):
#             if plot:
#                 axes[i].set_title(f"Ring {i+1} (q₀ = {q0_fixed:.4f} nm⁻¹): fit failed"); axes[i].axis('off')
#             strain_params.append(nan_dict.copy())
#             q_fit_vs_chi_list.append(np.full(n_bins, np.nan)) # Add NaNs for failed filt
#             logger.exception(f"Fit failed for Ring {i+1}")

#     if plot:
#         axes[-1].set_xlabel('Azimuth χ (°)')
#         fig.tight_layout()
#         fig_path = os.path.join(output_dir, "strain_vs_chi_plot_fitted.png")
#         fig.savefig(fig_path)
#         plt.close('all')
#         logger.info(f"Combined distortion fit plot saved to: {fig_path}")

#     # Calculate the average strain for each ring
#     strain_list = []
#     for i, row in enumerate(q_data):
#         mask = ~np.isnan(row)
#         q_avg = np.mean(row[mask]) if np.any(mask) else np.nan
#         q0 = strain_params[i].get('q0', np.nan)
#         if not np.isnan(q0) and q0 != 0 and not np.isnan(q_avg):
#             strain = (q0 - q_avg) / q0
#         else:
#             strain = np.nan
#         strain_list.append(strain)

#     # Extract the list of q0 values used for the fits
#     q0_list_out = [p.get('q0', np.nan) for p in strain_params]

#     # Save the full strain vs chi array
#     strain_vs_chi = (q0_chi_data - q_data) / q0_chi_data
#     strain_vs_chi_path = os.path.join(output_dir, "strain_vs_chi_peaks.txt")
#     np.savetxt(strain_vs_chi_path, strain_vs_chi, fmt="%.6e", delimiter="\t",
#                header="Rows = diffraction rings; Columns = azimuthal bins (strain vs chi data)")
#     logger.info(f"Strain vs chi centroid data saved to: {strain_vs_chi_path}")

#     # Save the fitted q0(chi) reference data
#     q_fit_vs_chi = np.array(q_fit_vs_chi_list)
#     q_fit_vs_chi_path = os.path.join(output_dir, "q0_vs_chi_FITTED.txt")
#     np.savetxt(q_fit_vs_chi_path, q_fit_vs_chi, fmt="%.6e", delimiter="\t",
#                header="Rows = diffraction rings; Columns = azimuthal bins (FITTED q0(chi) from strain model)")
#     logger.info(f"Fitted q0(chi) reference data saved to: {q_fit_vs_chi_path}")

#     return strain_params, strain_list, q0_list_out, strain_vs_chi_path


# if __name__ == '__main__':
#     main()