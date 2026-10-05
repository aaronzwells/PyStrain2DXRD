#!/usr/bin/env python3
"""
Convert /strain_fit/<phase>/<group> from a results HDF5 file into the
strain_tensor_summary.json layout read by generate_strain_maps_from_json():

    [ {"strain_tensor": [ {"q0":..., "eps_xx":..., ..., "eps_zz_err":...}, ...one dict per ring... ]},
      ... one entry per map pixel, in ROW-MAJOR order (row 0 first, columns fastest) ... ]

- A joint group (strain shape (n_frames, 6)) becomes ONE ring.
- A per-peak group (strain shape (n_frames, n_peaks, 6)) becomes one ring per peak
  (all peaks, or the subset given with --peaks).
- Strain convention is unchanged: eps = ln(q0/q) (positive = expansion), the same
  quantity your original fit function produced.
- Components that were not fitted are NaN, as before.

Frames are assumed to be acquired x-fast: consecutive frames run along a row, then
step to the next row (add --snake if every other row is scanned in reverse).
If the finished map is mirrored, add --flip-rows and/or --flip-cols.

Example:
    python hdf5_to_strain_json.py scan_fitresults.h5 --phase alumina --group 3comp_joint \
        --n-rows 44 --n-cols 8 --first-index 0
"""
import argparse
import json
import os

import h5py
import numpy as np

KEYS = ['eps_xx', 'eps_xy', 'eps_yy', 'eps_xz', 'eps_yz', 'eps_zz']


def grid_index(n_rows, n_cols, snake):
    """idx[row, col] = acquisition index k (0-based within the scan); x-fast acquisition."""
    k = np.arange(n_rows * n_cols)
    row, col = k // n_cols, k % n_cols      # frames run along a row, then step to the next row
    if snake:
        col = np.where(row % 2 == 1, n_cols - 1 - col, col)
    idx = np.empty((n_rows, n_cols), dtype=int)
    idx[row, col] = k
    return idx


def build_entries(strain, err, q0_ref, idx, first, peaks):
    if strain.ndim == 2:                                   # joint fit: a single ring
        strain, err = strain[:, None, :], err[:, None, :]
        finite = q0_ref[np.isfinite(q0_ref)]
        q0 = [float(finite.mean()) if finite.size else float('nan')]
    else:                                                  # per-peak fit: one ring per peak
        ring_ids = list(range(strain.shape[1])) if peaks is None else list(peaks)
        strain, err = strain[:, ring_ids], err[:, ring_ids]
        q0 = []
        for p in ring_ids:
            finite = q0_ref[p][np.isfinite(q0_ref[p])]
            q0.append(float(finite.mean()) if finite.size else float('nan'))

    entries = []
    for r in range(idx.shape[0]):
        for c in range(idx.shape[1]):
            f = first + int(idx[r, c])
            tensors = []
            for ri in range(strain.shape[1]):
                d = {'q0': q0[ri]}
                d.update({k: float(strain[f, ri, j]) for j, k in enumerate(KEYS)})
                d.update({k + '_err': float(err[f, ri, j]) for j, k in enumerate(KEYS)})
                tensors.append(d)
            entries.append({'frame_index': f, 'strain_tensor': tensors})
    return entries


def main(h5path, phase, group, n_rows, n_cols, snake=False, first_index=0,
         peaks=None, flip_rows=False, flip_cols=False, output=None):
    with h5py.File(h5path, 'r') as f:
        path = f'strain_fit/{phase}/{group}'
        if path not in f:
            avail = {p: list(f['strain_fit'][p].keys()) for p in f['strain_fit']} \
                if 'strain_fit' in f else {}
            raise KeyError(f"/{path} not found. Available: {avail}")
        g = f[path]
        strain, err, q0_ref = g['strain'][:], g['strain_err'][:], g['q0_ref'][:]
        ref_frame = g.attrs.get('ref_frame', None)

    n_needed = n_rows * n_cols
    if first_index + n_needed > strain.shape[0]:
        raise ValueError(f"Need frames {first_index}..{first_index + n_needed - 1} "
                         f"({n_rows}x{n_cols}) but the group has only {strain.shape[0]} frames")

    idx = grid_index(n_rows, n_cols, snake)
    if flip_rows:
        idx = idx[::-1]
    if flip_cols:
        idx = idx[:, ::-1]

    entries = build_entries(strain, err, q0_ref, idx, first_index, peaks)
    out = output or os.path.splitext(h5path)[0] + f'_{phase}_{group}_strain_tensor_summary.json'
    with open(out, 'w') as fh:
        json.dump(entries, fh)

    n_rings = len(entries[0]['strain_tensor'])
    print(f"Wrote {len(entries)} points x {n_rings} ring(s) to {out}")
    print(f"Frames {first_index}..{first_index + n_needed - 1} mapped to a {n_rows}x{n_cols} grid "
          f"(x-fast{', snake' if snake else ''}{', flip-rows' if flip_rows else ''}"
          f"{', flip-cols' if flip_cols else ''})")
    if ref_frame is not None:
        rel = int(ref_frame) - first_index
        hit = np.argwhere(idx == rel)
        if len(hit):
            print(f"Reference frame {int(ref_frame)} (zero-strain by construction) is at "
                  f"grid row {hit[0][0]}, col {hit[0][1]}")
        else:
            print(f"Reference frame {int(ref_frame)} lies outside the mapped frames")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('h5file')
    ap.add_argument('--phase', required=True)
    ap.add_argument('--group', required=True, help="e.g. 3comp, 3comp_joint, 5comp_joint")
    ap.add_argument('--n-rows', type=int, required=True)
    ap.add_argument('--n-cols', type=int, required=True)
    ap.add_argument('--snake', action='store_true', help="every other row is scanned in reverse")
    ap.add_argument('--first-index', type=int, default=0,
                    help="index in the HDF5 arrays of the first frame of the map")
    ap.add_argument('--peaks', type=int, nargs='+', default=None,
                    help="per-peak groups only: peaks to use as rings (default: all)")
    ap.add_argument('--flip-rows', action='store_true')
    ap.add_argument('--flip-cols', action='store_true')
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    main(a.h5file, a.phase, a.group, a.n_rows, a.n_cols, a.snake, a.first_index,
         a.peaks, a.flip_rows, a.flip_cols, a.output)
