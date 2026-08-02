"""WAXS peak fitting engine — fits diffraction peaks per eta bin across frames.
A. Chuang supplied 7/26/26

Usage:
    python waxs_peakfit.py data.h5 --config peaks.yaml [--output results.h5] [--dry-run] [--frames 0:100]
"""

import argparse
import os
import sys
import time

# Set GPU and compilation cache before JAX import
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/dev/shm/jax_cache')

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt

from waxs_peakfit_models import build_model, build_residual
from waxs_peakfit_config import load_config, expand_peaks, group_peaks


def parse_args():
    p = argparse.ArgumentParser(description='WAXS peak fitting engine')
    p.add_argument('h5file', help='Input HDF5 file with OmegaSumFrame')
    p.add_argument('--config', required=True, help='YAML config file')
    p.add_argument('--output', help='Output HDF5 file (default: <input>_fitresults.h5)')
    p.add_argument('--inplace', action='store_true', help='Save results into the input HDF5 file')
    p.add_argument('--dry-run', action='store_true', help='Show fit groups and exit')
    frame_group = p.add_mutually_exclusive_group()
    frame_group.add_argument('--frames', help='Frame range to include (e.g., 0:100, 0,5,10, all)')
    frame_group.add_argument('--skip', help='Frames to exclude (e.g., 0,5,10 or 0:3)')
    p.add_argument('--vis', nargs='?', type=float, const=1.0, default=None, metavar='N',
                   help='Visualize fits (hold N seconds per plot, default: 1)')
    p.add_argument('--gpu', type=int, default=None, metavar='ID',
                   help='GPU device ID to use (default: 0)')
    p.add_argument('--lineout-only', action='store_true',
                   help='Fit lineouts only, skip OmegaSumFrame fitting')
    p.add_argument('--pk', type=str, default=None,
                   help='Fit only selected peaks by index (e.g., 0,3,7)')
    return p.parse_args()


def parse_frames(spec, n_total):
    """Parse frame specification: '0:100', '0:100:2', '0,5,10', 'all'.

    Supports negative indices: -1 = last frame, -2 = second-to-last, etc.
    """
    def _resolve(idx):
        return idx if idx >= 0 else n_total + idx

    spec = spec.strip()
    if spec.lower() == 'all':
        return list(range(n_total))
    if ':' in spec:
        parts = spec.split(':')
        start = _resolve(int(parts[0])) if parts[0] else 0
        stop = _resolve(int(parts[1])) if len(parts) > 1 and parts[1] else n_total
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return list(range(start, stop, step))
    return [_resolve(int(x)) for x in spec.split(',')]


def load_radial_axis(f, config):
    """Load radial axis from HDF5 based on config radial_unit."""
    unit = config.get('radial_unit', 'Q')
    unit_map = {'R': 'R_map', 'TTH': 'TTh_map', 'Q': 'Q_map'}
    return f[f'geometry_maps/{unit_map[unit]}'][:, 0]


def _peak_search_bounds(peaks, x, y):
    """Compute non-overlapping search boundaries using valley finding between adjacent peaks."""
    centers = [pk['center'] for pk in peaks]
    splits = []
    for i in range(len(peaks) - 1):
        mid = (centers[i] + centers[i + 1]) / 2.0
        between = (x >= centers[i]) & (x <= centers[i + 1])
        if np.any(between):
            valley_idx = np.argmin(y[between])
            split = float(x[between][valley_idx])
        else:
            split = mid
        gap = centers[i + 1] - centers[i]
        margin = 0.25 * gap
        split = np.clip(split, centers[i] + margin, centers[i + 1] - margin)
        splits.append(split)

    bounds = []
    for i, pk in enumerate(peaks):
        lo = pk.get('roi_min', float(x[0]))
        hi = pk.get('roi_max', float(x[-1]))
        if i > 0:
            lo = max(lo, splits[i - 1])
        if i < len(peaks) - 1:
            hi = min(hi, splits[i])
        bounds.append((lo, hi))
    return bounds


def estimate_initial_params(x, y_median, peaks_in_group):
    """Estimate initial fitting parameters from median lineout."""
    bg_slope = float((y_median[-1] - y_median[0]) / (x[-1] - x[0] + 1e-30))
    bg_intercept = float(y_median[0] - bg_slope * x[0])

    search_bounds = _peak_search_bounds(peaks_in_group, x, y_median)

    params = []
    for pk, (slo, shi) in zip(peaks_in_group, search_bounds):
        pk_mask = (x >= slo) & (x <= shi)
        if np.any(pk_mask):
            max_idx = np.argmax(y_median[pk_mask])
            peak_y = float(y_median[pk_mask][max_idx])
            peak_x = float(x[pk_mask][max_idx])
        else:
            idx = np.argmin(np.abs(x - pk['center']))
            peak_y = float(y_median[idx])
            peak_x = float(x[idx])
        bg_at_peak = bg_slope * peak_x + bg_intercept
        amp = max(peak_y - bg_at_peak, 1.0)
        half_max = bg_at_peak + amp / 2.0
        pk_y = y_median[pk_mask] if np.any(pk_mask) else y_median
        pk_x = x[pk_mask] if np.any(pk_mask) else x
        above = np.where(pk_y > half_max)[0]
        if len(above) >= 2:
            width = float(pk_x[above[-1]] - pk_x[above[0]])
            width = max(width, float(np.mean(np.diff(pk_x))))
        else:
            width = float(pk_x[-1] - pk_x[0]) / 20.0
        params.extend([amp, peak_x, width, 0.5])
    params.extend([bg_slope, bg_intercept])
    return jnp.array(params)


def compute_skip_mask(data_2d, config):
    """Compute skip mask for eta bins. True = skip this bin."""
    skip_cfg = config.get('skip', {})
    min_intensity = skip_cfg.get('min_max_intensity', 0)
    min_snr = skip_cfg.get('min_snr', 0)

    max_vals = np.max(data_2d, axis=1)
    mask = np.zeros(data_2d.shape[0], dtype=bool)

    if min_intensity > 0:
        mask |= max_vals < min_intensity
    if min_snr > 0:
        median_vals = np.median(data_2d, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = max_vals / np.where(median_vals > 0, median_vals, 1)
        mask |= snr < min_snr
    return mask


def _update_vis(fig, ax_top, ax_bot, x, y, init_p, fit_p, model_fn,
                group, frame_idx, eta_idx, eta_val, chi2_val):
    y_init = np.asarray(model_fn(jnp.array(init_p), jnp.array(x)))
    y_fit = np.asarray(model_fn(jnp.array(fit_p), jnp.array(x)))
    residual = y - y_fit

    ax_top.clear()
    ax_top.plot(x, y, 'o', ms=3, alpha=0.5, color='blue', label='Data')
    ax_top.plot(x, y_init, '--', color='gray', alpha=0.7, label='Init guess')
    ax_top.plot(x, y_fit, '-', color='red', linewidth=2, label='Fit')

    print(f"    --- Frame {frame_idx}, Eta[{eta_idx}] = {eta_val:.1f}° ---")
    print(f"    {'peak':>12s}  {'':>6s}  {'Amp':>10s}  {'Center':>16s}  {'Width':>16s}  {'Eta':>12s}")
    for pi, pk in enumerate(group['peaks']):
        label = f"{pk['phase']}:{pk['peak_idx']}"
        print(f"    {label:>12s}  {'init':>6s}  {init_p[4*pi]:10.1f}  {init_p[4*pi+1]:16.4f}  {init_p[4*pi+2]:16.4f}  {init_p[4*pi+3]:12.2f}")
        print(f"    {'':>12s}  {'fit':>6s}  {fit_p[4*pi]:10.1f}  {fit_p[4*pi+1]:16.4f}  {fit_p[4*pi+2]:16.4f}  {fit_p[4*pi+3]:12.2f}")
    print(f"    {'bg':>12s}  {'init':>6s}  slope={init_p[-2]:.4f}  intercept={init_p[-1]:.1f}")
    print(f"    {'':>12s}  {'fit':>6s}  slope={fit_p[-2]:.4f}  intercept={fit_p[-1]:.1f}")
    print(f"    chi2={chi2_val:.2f}")

    lines = []
    for pi, pk in enumerate(group['peaks']):
        lines.append(f"{pk['phase']}:{pk['peak_idx']}  "
                     f"A={fit_p[4*pi]:.1f} C={fit_p[4*pi+1]:.4f} "
                     f"W={fit_p[4*pi+2]:.4f} η={fit_p[4*pi+3]:.3f}")
    lines.append(f"χ²={chi2_val:.1f}")
    ax_top.text(0.98, 0.97, '\n'.join(lines), transform=ax_top.transAxes,
                va='top', ha='right', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax_top.set_yscale('log')
    ax_top.set_ylabel('Intensity')
    ax_top.set_title(f'Frame {frame_idx}, Eta[{eta_idx}] = {eta_val:.1f}°')
    ax_top.legend(loc='upper left')
    ax_top.grid(alpha=0.3)

    ax_bot.clear()
    ax_bot.plot(x, residual, 'o', ms=2, color='green')
    ax_bot.axhline(0, color='gray', linestyle='--')
    ax_bot.set_xlabel('Radial')
    ax_bot.set_ylabel('Residual')
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def fit_group(omega_ds, radial_axis, group, config, frame_indices,
              vis_hold=None, eta_axis=None, padded_size=None,
              lineout_params=None):
    """Fit one group across all frames and eta bins."""
    n_peaks = group['n_peaks']
    roi_mask = (radial_axis >= group['roi_min']) & (radial_axis <= group['roi_max'])
    radial_scale = config.get('radial_scale', 1.0)
    x_raw = radial_axis[roi_mask] * radial_scale
    n_roi = int(roi_mask.sum())
    n_eta = omega_ds.shape[2]
    n_frames = len(frame_indices)
    n_params = 4 * n_peaks + 2

    # Pad x to common size so all groups with same n_peaks share one compiled kernel
    if padded_size is not None and padded_size > n_roi:
        pad_n = padded_size - n_roi
        x_padded = np.concatenate([x_raw, np.full(pad_n, x_raw[-1])])
        w = jnp.array(np.concatenate([np.ones(n_roi), np.zeros(pad_n)]))
    else:
        x_padded = x_raw
        w = jnp.ones(n_roi)
        padded_size = n_roi
    x = jnp.array(x_padded)

    # Build model and solver (compiled once per n_peaks + padded_size)
    model_fn = build_model(n_peaks)

    def weighted_residual(params, x, y):
        return (model_fn(params, x) - y) * w

    solver = LevenbergMarquardt(weighted_residual, maxiter=200)

    @jax.jit
    def fit_one(init_params, y):
        result = solver.run(init_params, x, y)
        return result.params

    fit_batch = jax.jit(jax.vmap(fit_one, in_axes=(0, 0)))

    @jax.jit
    def errors_one(params, y):
        J = jax.jacobian(lambda p: model_fn(p, x))(params)
        r = model_fn(params, x) - y
        J = J * w[:, None]
        r = r * w
        n_dof = jnp.maximum(jnp.sum(w) - params.shape[0], 1)
        chi2 = jnp.sum(r**2)
        cov = jnp.linalg.inv(J.T @ J + 1e-10 * jnp.eye(n_params)) * (chi2 / n_dof)
        return jnp.sqrt(jnp.abs(jnp.diag(cov))), chi2

    errors_batch = jax.jit(jax.vmap(errors_one, in_axes=(0, 0)))

    # Allocate storage
    all_params = np.full((n_frames, n_eta, n_params), np.nan)
    all_errors = np.full((n_frames, n_eta, n_params), np.nan)
    all_chi2 = np.full((n_frames, n_eta), np.nan)
    all_mask = np.zeros((n_frames, n_eta), dtype=bool)

    drift_cfg = config.get('drift', {})
    drift_enabled = drift_cfg.get('enable', False)
    max_shift = drift_cfg.get('max_shift', 0.5)
    drift_flags = np.zeros((n_frames, n_peaks), dtype=bool)
    config_centers = np.array([pk['center'] for pk in group['peaks']])
    if radial_scale != 1.0:
        scaled_peaks = [dict(pk, center=pk['center'] * radial_scale,
                             roi_min=pk['roi_min'] * radial_scale,
                             roi_max=pk['roi_max'] * radial_scale)
                        for pk in group['peaks']]
    else:
        scaled_peaks = group['peaks']

    if vis_hold is not None:
        import matplotlib.pyplot as plt
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6),
                                              height_ratios=[3, 1])
        plt.ion()
        plt.show()

    # Warm-up JIT compilation on first call
    pad_info = f", padded={padded_size}" if padded_size > n_roi else ""
    print(f"  Compiling (n_peaks={n_peaks}, n_roi={n_roi}{pad_info})...", end=' ', flush=True)
    t_compile = time.time()

    for fi, frame_idx in enumerate(frame_indices):
        frame_data = omega_ds[frame_idx][roi_mask, :]  # (n_roi, n_eta)
        if padded_size > n_roi:
            frame_data = np.concatenate(
                [frame_data, np.zeros((padded_size - n_roi, n_eta))], axis=0)
        y_all = jnp.array(frame_data.T)  # (n_eta, padded_size)

        # Skip mask — use unpadded data
        skip = compute_skip_mask(np.asarray(y_all[:, :n_roi]), config)
        all_mask[fi] = ~skip

        # Initial guess (per-eta) — use unpadded data
        x_np = np.asarray(x[:n_roi])
        y_np = np.asarray(y_all[:, :n_roi])
        if fi > 0 and np.any(all_mask[fi - 1]):
            valid_prev = all_params[fi - 1][all_mask[fi - 1]]
            median_p = np.median(valid_prev, axis=0)
            init_all = np.tile(median_p, (n_eta, 1))
            prev_valid = all_mask[fi - 1]
            init_all[prev_valid] = all_params[fi - 1][prev_valid]
            if lineout_params is not None:
                init_all[~prev_valid] = lineout_params[frame_idx]
            if drift_enabled:
                for pi in range(n_peaks):
                    fitted_cen = float(median_p[4 * pi + 1])
                    if abs(fitted_cen - config_centers[pi]) > max_shift:
                        drift_flags[fi, pi] = True
                        init_all[:, 4 * pi + 1] = config_centers[pi]
            if radial_scale != 1.0:
                for pi in range(n_peaks):
                    init_all[:, 4*pi+1] *= radial_scale
                    init_all[:, 4*pi+2] *= radial_scale
                init_all[:, -2] /= radial_scale
            # Override centers with per-eta data max positions
            y_med = np.median(y_np, axis=0)
            search_bounds = _peak_search_bounds(scaled_peaks, x_np, y_med)
            for pi, (slo, shi) in enumerate(search_bounds):
                pk_mask = (x_np >= slo) & (x_np <= shi)
                if np.any(pk_mask):
                    for ei in range(n_eta):
                        max_idx = np.argmax(y_np[ei, pk_mask])
                        init_all[ei, 4*pi+1] = float(x_np[pk_mask][max_idx])
        elif lineout_params is not None:
            init_all = np.tile(lineout_params[frame_idx], (n_eta, 1))
            if radial_scale != 1.0:
                for pi in range(n_peaks):
                    init_all[:, 4*pi+1] *= radial_scale
                    init_all[:, 4*pi+2] *= radial_scale
                init_all[:, -2] /= radial_scale
            y_med = np.median(y_np, axis=0)
            search_bounds = _peak_search_bounds(scaled_peaks, x_np, y_med)
            for pi, (slo, shi) in enumerate(search_bounds):
                pk_mask = (x_np >= slo) & (x_np <= shi)
                if np.any(pk_mask):
                    for ei in range(n_eta):
                        max_idx = np.argmax(y_np[ei, pk_mask])
                        init_all[ei, 4*pi+1] = float(x_np[pk_mask][max_idx])
        else:
            init_all = np.zeros((n_eta, n_params))
            for ei in range(n_eta):
                init_all[ei] = estimate_initial_params(x_np, y_np[ei], scaled_peaks)
        init_p = jnp.array(init_all)

        # Fit all eta bins
        fitted = fit_batch(init_p, y_all)
        fitted_np = np.array(fitted)

        # Post-hoc constraints: amp>0, width>0, eta in [0,1]
        for pi in range(n_peaks):
            fitted_np[:, 4*pi] = np.abs(fitted_np[:, 4*pi])
            fitted_np[:, 4*pi+2] = np.abs(fitted_np[:, 4*pi+2])
            fitted_np[:, 4*pi+3] = np.clip(fitted_np[:, 4*pi+3], 0, 1)

        # Errors and chi2
        errs, chi2 = errors_batch(fitted, y_all)
        all_chi2[fi] = np.asarray(chi2)

        if fi == 0:
            print(f"done ({time.time() - t_compile:.1f}s)")

        if (fi + 1) % 200 == 0 or fi == n_frames - 1:
            print(f"  Frame {fi + 1}/{n_frames} (#{frame_idx})")

        if vis_hold is not None:
            for ei in range(n_eta):
                if not all_mask[fi, ei]:
                    continue
                _update_vis(fig, ax_top, ax_bot,
                           np.array(x[:n_roi]), np.array(y_all[ei, :n_roi]),
                           np.array(init_p[ei]), fitted_np[ei],
                           model_fn, group, frame_idx, ei,
                           float(eta_axis[ei]) if eta_axis is not None else float(ei),
                           float(all_chi2[fi, ei]))
                if vis_hold < 0:
                    key = [None]
                    def _on_key(event):
                        key[0] = event.key
                    cid = fig.canvas.mpl_connect('key_press_event', _on_key)
                    plt.waitforbuttonpress()
                    if plt.fignum_exists(fig.number):
                        fig.canvas.mpl_disconnect(cid)
                    if key[0] in ('escape', 'q') or not plt.fignum_exists(fig.number):
                        if plt.fignum_exists(fig.number):
                            plt.ioff()
                            plt.close(fig)
                        print("  Visualization stopped by user.")
                        return all_params, all_errors, all_chi2, all_mask, drift_flags
                else:
                    plt.pause(vis_hold)

        # Unscale fitted params and errors back to original units
        if radial_scale != 1.0:
            for pi in range(n_peaks):
                fitted_np[:, 4*pi+1] /= radial_scale
                fitted_np[:, 4*pi+2] /= radial_scale
            fitted_np[:, -2] *= radial_scale
            errs_np = np.asarray(errs)
            for pi in range(n_peaks):
                errs_np[:, 4*pi+1] /= radial_scale
                errs_np[:, 4*pi+2] /= radial_scale
            errs_np[:, -2] *= radial_scale
            all_errors[fi] = errs_np
        else:
            all_errors[fi] = np.asarray(errs)
        all_params[fi] = fitted_np

        # Clear skipped eta bins to NaN
        if np.any(skip):
            all_params[fi, skip] = np.nan
            all_errors[fi, skip] = np.nan
            all_chi2[fi, skip] = np.nan

    if vis_hold is not None:
        plt.ioff()
        plt.close(fig)

    return all_params, all_errors, all_chi2, all_mask, drift_flags


def fit_group_lineout(lineout_ds, radial_axis, group, config, frame_indices,
                      vis_hold=None, padded_size=None):
    """Fit one group across all frames on lineout (1D) data."""
    n_peaks = group['n_peaks']
    roi_mask = (radial_axis >= group['roi_min']) & (radial_axis <= group['roi_max'])
    radial_scale = config.get('radial_scale', 1.0)
    x_raw = radial_axis[roi_mask] * radial_scale
    n_roi = int(roi_mask.sum())
    n_frames = len(frame_indices)
    n_params = 4 * n_peaks + 2

    if padded_size is not None and padded_size > n_roi:
        pad_n = padded_size - n_roi
        x_padded = np.concatenate([x_raw, np.full(pad_n, x_raw[-1])])
        w = jnp.array(np.concatenate([np.ones(n_roi), np.zeros(pad_n)]))
    else:
        x_padded = x_raw
        w = jnp.ones(n_roi)
        padded_size = n_roi
    x = jnp.array(x_padded)

    model_fn = build_model(n_peaks)

    def weighted_residual(params, x, y):
        return (model_fn(params, x) - y) * w

    solver = LevenbergMarquardt(weighted_residual, maxiter=200)

    @jax.jit
    def fit_one(init_params, y):
        result = solver.run(init_params, x, y)
        return result.params

    @jax.jit
    def errors_one(params, y):
        J = jax.jacobian(lambda p: model_fn(p, x))(params)
        r = model_fn(params, x) - y
        J = J * w[:, None]
        r = r * w
        n_dof = jnp.maximum(jnp.sum(w) - params.shape[0], 1)
        chi2 = jnp.sum(r**2)
        cov = jnp.linalg.inv(J.T @ J + 1e-10 * jnp.eye(n_params)) * (chi2 / n_dof)
        return jnp.sqrt(jnp.abs(jnp.diag(cov))), chi2

    all_params = np.full((n_frames, n_params), np.nan)
    all_errors = np.full((n_frames, n_params), np.nan)
    all_chi2 = np.full(n_frames, np.nan)

    if radial_scale != 1.0:
        scaled_peaks = [dict(pk, center=pk['center'] * radial_scale,
                             roi_min=pk['roi_min'] * radial_scale,
                             roi_max=pk['roi_max'] * radial_scale)
                        for pk in group['peaks']]
    else:
        scaled_peaks = group['peaks']

    if vis_hold is not None:
        import matplotlib.pyplot as plt
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6),
                                              height_ratios=[3, 1])
        plt.ion()
        plt.show()

    pad_info = f", padded={padded_size}" if padded_size > n_roi else ""
    print(f"  Compiling (n_peaks={n_peaks}, n_roi={n_roi}{pad_info})...", end=' ', flush=True)
    t_compile = time.time()

    x_np = np.asarray(x[:n_roi])

    for fi, frame_idx in enumerate(frame_indices):
        y_raw = lineout_ds[frame_idx][roi_mask]
        y_np = np.asarray(y_raw)

        if compute_skip_mask(y_np[np.newaxis, :], config)[0]:
            if fi == 0:
                print(f"done ({time.time() - t_compile:.1f}s)")
            if (fi + 1) % 200 == 0 or fi == n_frames - 1:
                print(f"  Frame {fi + 1}/{n_frames} (#{frame_idx}) — skipped")
            continue

        if padded_size > n_roi:
            y_raw = np.concatenate([y_raw, np.zeros(padded_size - n_roi)])
        y = jnp.array(y_raw)

        init_p = estimate_initial_params(x_np, y_np, scaled_peaks)

        fitted = fit_one(init_p, y)
        fitted_np = np.array(fitted)

        for pi in range(n_peaks):
            fitted_np[4*pi] = abs(fitted_np[4*pi])
            fitted_np[4*pi+2] = abs(fitted_np[4*pi+2])
            fitted_np[4*pi+3] = np.clip(fitted_np[4*pi+3], 0, 1)

        errs, chi2 = errors_one(fitted, y)

        if fi == 0:
            print(f"done ({time.time() - t_compile:.1f}s)")

        if (fi + 1) % 200 == 0 or fi == n_frames - 1:
            print(f"  Frame {fi + 1}/{n_frames} (#{frame_idx})")

        if vis_hold is not None:
            _update_vis(fig, ax_top, ax_bot,
                       np.array(x[:n_roi]), y_np,
                       np.array(init_p), fitted_np,
                       model_fn, group, frame_idx, -1, 0.0,
                       float(chi2))
            ax_top.set_title(f'Lineout — Frame {frame_idx}')
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            if vis_hold < 0:
                key = [None]
                def _on_key(event):
                    key[0] = event.key
                cid = fig.canvas.mpl_connect('key_press_event', _on_key)
                plt.waitforbuttonpress()
                if plt.fignum_exists(fig.number):
                    fig.canvas.mpl_disconnect(cid)
                if key[0] in ('escape', 'q') or not plt.fignum_exists(fig.number):
                    if plt.fignum_exists(fig.number):
                        plt.ioff()
                        plt.close(fig)
                    print("  Visualization stopped by user.")
                    return all_params, all_errors, all_chi2
            else:
                plt.pause(vis_hold)

        if radial_scale != 1.0:
            for pi in range(n_peaks):
                fitted_np[4*pi+1] /= radial_scale
                fitted_np[4*pi+2] /= radial_scale
            fitted_np[-2] *= radial_scale
            errs_np = np.asarray(errs)
            for pi in range(n_peaks):
                errs_np[4*pi+1] /= radial_scale
                errs_np[4*pi+2] /= radial_scale
            errs_np[-2] *= radial_scale
            all_errors[fi] = errs_np
        else:
            all_errors[fi] = np.asarray(errs)
        all_params[fi] = fitted_np
        all_chi2[fi] = float(chi2)

    if vis_hold is not None:
        plt.ioff()
        plt.close(fig)

    return all_params, all_errors, all_chi2


def _collect_phase_data(groups, results, config, n_frames, n_eta,
                        frame_indices=None):
    """Redistribute group fit results into per-phase arrays."""
    phases = {}
    for entry in config['peaks']:
        phase = entry['phase']
        n_pk = len(entry['center'])
        phases[phase] = {
            'n_peaks': n_pk,
            'params': np.full((n_frames, n_pk, n_eta, 4), np.nan),
            'errors': np.full((n_frames, n_pk, n_eta, 4), np.nan),
            'background': np.full((n_frames, n_pk, n_eta, 2), np.nan),
            'chi2': np.full((n_frames, n_pk, n_eta), np.nan),
            'mask': np.zeros((n_frames, n_pk, n_eta), dtype=bool),
            'drift_flag': np.zeros((n_frames, n_pk), dtype=bool),
        }

    idx = frame_indices if frame_indices is not None else slice(None)
    for group, (params, errors, chi2, mask, drift_flags) in zip(groups, results):
        for pi, pk in enumerate(group['peaks']):
            ph = phases[pk['phase']]
            pidx = pk['peak_idx']
            ph['params'][idx, pidx, :, :] = params[:, :, 4*pi:4*pi+4]
            ph['errors'][idx, pidx, :, :] = errors[:, :, 4*pi:4*pi+4]
            ph['background'][idx, pidx, :, :] = params[:, :, -2:]
            ph['chi2'][idx, pidx, :] = chi2
            ph['mask'][idx, pidx, :] = mask
            ph['drift_flag'][idx, pidx] = drift_flags[:, pi]
    return phases


def _collect_lineout_data(groups, results, config, n_frames,
                          frame_indices=None):
    """Redistribute lineout fit results into per-phase arrays."""
    phases = {}
    for entry in config['peaks']:
        phase = entry['phase']
        n_pk = len(entry['center'])
        phases[phase] = {
            'n_peaks': n_pk,
            'params': np.full((n_frames, n_pk, 4), np.nan),
            'errors': np.full((n_frames, n_pk, 4), np.nan),
            'background': np.full((n_frames, n_pk, 2), np.nan),
            'chi2': np.full((n_frames, n_pk), np.nan),
        }

    idx = frame_indices if frame_indices is not None else slice(None)
    for group, (params, errors, chi2) in zip(groups, results):
        for pi, pk in enumerate(group['peaks']):
            ph = phases[pk['phase']]
            pidx = pk['peak_idx']
            ph['params'][idx, pidx, :] = params[:, 4*pi:4*pi+4]
            ph['errors'][idx, pidx, :] = errors[:, 4*pi:4*pi+4]
            ph['background'][idx, pidx, :] = params[:, -2:]
            ph['chi2'][idx, pidx] = chi2
    return phases


def _write_lineout_results(f, groups, results, config, n_frames,
                           frame_indices=None):
    """Write lineout fit results into an open HDF5 file handle."""
    phases = _collect_lineout_data(groups, results, config, n_frames,
                                   frame_indices)

    if 'fit_results_lineout' in f:
        del f['fit_results_lineout']
    fr = f.create_group('fit_results_lineout')
    for phase, data in phases.items():
        g = fr.create_group(phase)
        for key in ['params', 'errors', 'background', 'chi2']:
            g.create_dataset(key, data=data[key])

    meta = fr.create_group('metadata')
    meta.attrs['model'] = config.get('model', 'pseudo_voigt')
    meta.attrs['radial_unit'] = config.get('radial_unit', 'Q')
    meta.attrs['radial_scale'] = config.get('radial_scale', 1.0)
    import yaml
    meta.attrs['config_yaml'] = yaml.dump(config, default_flow_style=False)


def _write_fit_results(f, groups, results, config, n_frames, n_eta, eta_axis,
                       frame_indices=None):
    """Write fit results into an open HDF5 file handle."""
    phases = _collect_phase_data(groups, results, config, n_frames, n_eta,
                                 frame_indices)

    if 'fit_results' in f:
        del f['fit_results']
    fr = f.create_group('fit_results')
    for phase, data in phases.items():
        g = fr.create_group(phase)
        for key in ['params', 'errors', 'background', 'chi2', 'mask', 'drift_flag']:
            g.create_dataset(key, data=data[key])

    fr.create_dataset('eta', data=eta_axis)

    meta = fr.create_group('metadata')
    meta.attrs['model'] = config.get('model', 'pseudo_voigt')
    meta.attrs['radial_unit'] = config.get('radial_unit', 'Q')
    meta.attrs['radial_scale'] = config.get('radial_scale', 1.0)
    import yaml
    meta.attrs['config_yaml'] = yaml.dump(config, default_flow_style=False)
    for gi, group in enumerate(groups):
        gm = meta.create_group(f'group_{gi}')
        gm.attrs['roi_min'] = group['roi_min']
        gm.attrs['roi_max'] = group['roi_max']
        for pi, pk in enumerate(group['peaks']):
            gm.attrs[f'peak_{pi}'] = f"{pk['phase']}:{pk['peak_idx']}"


def main():
    args = parse_args()
    config = load_config(args.config)

    all_peaks = expand_peaks(config)

    if args.pk is not None:
        pk_indices = set(int(x) for x in args.pk.split(','))
        peaks = [pk for i, pk in enumerate(all_peaks) if i in pk_indices]
        if not peaks:
            print(f"No peaks matched --pk {args.pk} (total: {len(all_peaks)})")
            sys.exit(1)
        print(f"Selected {len(peaks)}/{len(all_peaks)} peaks: {sorted(pk_indices)}")
    else:
        peaks = all_peaks

    groups = group_peaks(peaks)

    print(f"Config: {len(peaks)} peaks -> {len(groups)} fit groups")
    for gi, g in enumerate(groups):
        names = [f"{p['phase']}:{p['peak_idx']}" for p in g['peaks']]
        print(f"  Group {gi}: [{g['roi_min']:.4f}, {g['roi_max']:.4f}] "
              f"({g['n_peaks']} peak{'s' if g['n_peaks'] > 1 else ''}) — {', '.join(names)}")

    if args.dry_run:
        return

    f = h5py.File(args.h5file, 'a' if args.inplace else 'r')
    radial_axis = load_radial_axis(f, config)
    eta_axis = f['geometry_maps/Eta_map'][0, :]
    omega_ds = f['OmegaSumFrame']
    n_frames_total, _n_rad, n_eta = omega_ds.shape
    lineout_ds = f['lineouts']

    print(f"Radial unit: {config.get('radial_unit', 'Q')}")
    if config.get('radial_scale', 1.0) != 1.0:
        print(f"Radial scale: {config['radial_scale']}x")

    # Compute padded ROI sizes — one compiled kernel per n_peaks
    roi_sizes = {}
    for group in groups:
        n_pk = group['n_peaks']
        n_roi = int(((radial_axis >= group['roi_min']) &
                     (radial_axis <= group['roi_max'])).sum())
        roi_sizes.setdefault(n_pk, []).append(n_roi)
    padded_map = {n_pk: max(sizes) for n_pk, sizes in roi_sizes.items()}

    # --- Determine skip set ---
    skip_set = set()
    if args.skip:
        skip_set = set(parse_frames(args.skip, n_frames_total))
        print(f"Skipping {len(skip_set)} frames: {sorted(skip_set)}")

    # --- Determine frame indices ---
    if args.frames:
        frame_indices = parse_frames(args.frames, n_frames_total)
        frame_indices = [i for i in frame_indices if i not in skip_set]
    else:
        frame_indices = [i for i in range(n_frames_total) if i not in skip_set]

    # --- Stage 1: Fit lineouts ---
    all_frame_indices = frame_indices
    print(f"\n=== Stage 1: Fitting lineouts ({len(all_frame_indices)} frames) ===")
    lineout_results = []
    t_lineout = time.time()
    for gi, group in enumerate(groups):
        names = [f"{p['phase']}:{p['peak_idx']}" for p in group['peaks']]
        print(f"\nGroup {gi}: {', '.join(names)}")
        t0 = time.time()
        result = fit_group_lineout(lineout_ds, radial_axis, group, config,
                                   all_frame_indices, vis_hold=args.vis,
                                   padded_size=padded_map[group['n_peaks']])
        print(f"  Elapsed: {time.time() - t0:.1f}s")
        lineout_results.append(result)
    print(f"\nLineout fitting time: {time.time() - t_lineout:.1f}s")

    # --- Stage 2: Fit OmegaSumFrame (unless --lineout-only) ---
    omega_results = None
    if not args.lineout_only:
        n_frames = len(frame_indices)

        print(f"\n=== Stage 2: Fitting OmegaSumFrame ({n_frames} frames x {n_eta} eta bins) ===")
        omega_results = []
        t_omega = time.time()
        for gi, group in enumerate(groups):
            names = [f"{p['phase']}:{p['peak_idx']}" for p in group['peaks']]
            print(f"\nGroup {gi}: {', '.join(names)}")
            t0 = time.time()
            lo_params = lineout_results[gi][0]
            result = fit_group(omega_ds, radial_axis, group, config, frame_indices,
                               vis_hold=args.vis, eta_axis=eta_axis,
                               padded_size=padded_map[group['n_peaks']],
                               lineout_params=lo_params)
            print(f"  Elapsed: {time.time() - t0:.1f}s")
            omega_results.append(result)
        print(f"\nOmegaSumFrame fitting time: {time.time() - t_omega:.1f}s")

    # --- Save results ---
    lo_idx = all_frame_indices if len(all_frame_indices) < n_frames_total else None
    if not args.lineout_only:
        om_idx = frame_indices if len(frame_indices) < n_frames_total else None
    else:
        om_idx = None
    if args.inplace:
        output = args.h5file
        _write_lineout_results(f, groups, lineout_results, config,
                               n_frames_total, lo_idx)
        if omega_results is not None:
            _write_fit_results(f, groups, omega_results, config,
                               n_frames_total, n_eta, eta_axis, om_idx)
        f.close()
    else:
        f.close()
        output = args.output or args.h5file.replace('.h5', '_fitresults.h5')
        with h5py.File(output, 'w') as fout:
            _write_lineout_results(fout, groups, lineout_results, config,
                                   n_frames_total, lo_idx)
            if omega_results is not None:
                _write_fit_results(fout, groups, omega_results, config,
                                   n_frames_total, n_eta, eta_axis, om_idx)
    print(f"Results saved to {output}")

    # Summary
    if omega_results is not None:
        for group, (params, errors, chi2, mask, drift_flags) in zip(groups, omega_results):
            names = [f"{p['phase']}:{p['peak_idx']}" for p in group['peaks']]
            fitted = mask.sum()
            skipped = (~mask).sum()
            drifted = drift_flags.sum()
            print(f"  {', '.join(names)}: {fitted} fitted, {skipped} skipped"
                  + (f", {drifted} drift warnings" if drifted > 0 else ""))


if __name__ == '__main__':
    main()
