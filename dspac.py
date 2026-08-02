#!/usr/bin/env python3
"""List allowed crystallographic reflections with d-spacings and q values.

Given a space group and lattice parameters, enumerates all symmetry-allowed
{hkl} families, sorted by d-spacing. Systematic absences are handled by
pymatgen's XRDCalculator using a dummy structure.

Requires: pymatgen (pip install pymatgen)

Authors: Claude, CPC
"""

import sys
import argparse
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# (n_params, description) per crystal system
_CRYSTAL_PARAMS = {
    'cubic':        (1, 'a'),
    'hexagonal':    (2, 'a,c'),
    'trigonal':     (2, 'a,c'),
    'tetragonal':   (2, 'a,c'),
    'orthorhombic': (3, 'a,b,c'),
    'monoclinic':   (4, 'a,b,c,beta(deg)'),
    'triclinic':    (6, 'a,b,c,alpha,beta,gamma(deg)'),
}


def _make_lattice(crystal_system, p):
    """Build a pymatgen Lattice from crystal system and parameter list."""
    if crystal_system == 'cubic':
        return Lattice.cubic(p[0])
    if crystal_system in ('hexagonal', 'trigonal'):
        return Lattice.hexagonal(p[0], p[1])
    if crystal_system == 'tetragonal':
        return Lattice.tetragonal(p[0], p[1])
    if crystal_system == 'orthorhombic':
        return Lattice.orthorhombic(*p[:3])
    if crystal_system == 'monoclinic':
        return Lattice.monoclinic(*p[:4])
    if crystal_system == 'triclinic':
        return Lattice(*p[:6])
    raise ValueError(f"Unknown crystal system: {crystal_system}")


def main():
    ap = argparse.ArgumentParser(
        description='List d-spacings for a crystal structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  %(prog)s Fm-3m 5.411            # CeO2 (cubic)\n'
               '  %(prog)s 225 5.411              # same, by number\n'
               '  %(prog)s Im-3m 2.8665           # BCC Fe (229)\n'
               '  %(prog)s 221 4.156826           # LaB6\n'
               '  %(prog)s P6_3/mmc 3.21,5.21     # Ti (hexagonal)\n'
               '  %(prog)s 62 4.94,5.40,7.41      # Fe2O3 (orthorhombic)\n')
    ap.add_argument('spacegroup',
                    help='Space group symbol (e.g. Fm-3m) or number (e.g. 225)')
    ap.add_argument('lattice',
                    help='Lattice parameter(s) in Angstrom, comma-separated '
                         '(cubic: a | hex/tet: a,c | ortho: a,b,c | '
                         'mono: a,b,c,beta | tri: a,b,c,alpha,beta,gamma)')
    ap.add_argument('--npk', type=int, default=15,
                    help='Number of peaks to list (default: 15)')
    ap.add_argument('--min-d', type=float, default=None,
                    help='Minimum d-spacing in Angstrom (overrides --npk when given)')
    ap.add_argument('--lambda', type=float, default=71.676, dest='lam',
                    help='X-ray wavelength in Angstrom, or energy in keV if > 3 '
                         '(default: 71.676 keV)')
    args = ap.parse_args()

    # --- wavelength / energy ---
    HC_OVER_E = 12.398419057638671  # hc in keV*Angstrom
    if args.lam > 3.0:
        energy_keV = args.lam
        user_lambda = HC_OVER_E / energy_keV
    else:
        energy_keV = HC_OVER_E / args.lam
        user_lambda = args.lam

    # --- space group ---
    try:
        sg = SpaceGroup.from_int_number(int(args.spacegroup))
    except ValueError:
        try:
            sg = SpaceGroup(args.spacegroup)
        except ValueError as e:
            print(f"Error: invalid space group '{args.spacegroup}': {e}")
            sys.exit(1)

    crystal_system = sg.crystal_system
    n_needed, names = _CRYSTAL_PARAMS[crystal_system]

    # --- lattice parameters ---
    params = [float(x) for x in args.lattice.split(',')]
    if len(params) < n_needed:
        print(f"Error: {crystal_system} system needs {n_needed} parameter(s): {names}")
        sys.exit(1)

    lattice = _make_lattice(crystal_system, params)

    # --- build dummy structure (atom at general position to avoid
    #     accidental extinctions from special-position symmetry) ---
    structure = Structure.from_spacegroup(
        sg.int_number, lattice, ['Si'], [[0.12, 0.23, 0.34]])

    # --- use XRDCalculator to get allowed reflections ---
    # Short wavelength ensures all reflections are accessible
    min_d = args.min_d if args.min_d is not None else 0.5
    wavelength = max(2 * min_d * 0.99, 0.01)
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(0.5, 179.5))

    # --- collect results ---
    rows = []
    for hkl_entries, d in zip(pattern.hkls, pattern.d_hkls):
        if args.min_d is not None and d < args.min_d:
            continue
        q = 2 * np.pi / d
        labels = []
        for entry in hkl_entries:
            h, k, l = entry['hkl']
            labels.append(f'{{{h} {k} {l}}}')
        rows.append((' / '.join(labels), d, q))

    # sort by d-spacing descending, then truncate to npk unless --min-d given
    rows.sort(key=lambda r: -r[1])
    if args.min_d is None:
        rows = rows[:args.npk]

    if not rows:
        print('No allowed reflections found.')
        sys.exit(0)

    # --- output ---
    print(f'Space group : {sg.symbol} (#{sg.int_number}), {crystal_system}')
    print(f'Lattice     : {", ".join(f"{p:.4f}" for p in params)} Angstrom')
    print(f'Energy      : {energy_keV:.3f} keV (lambda = {user_lambda:.6f} A)')
    print()

    hdr = f'{"#":>3}  {"hkl":<28}  {"d (A)":>10}  {"q (1/A)":>10}  {"2th (deg)":>10}'
    print(hdr)
    print('-' * len(hdr))
    for i, (hkl_str, d, q) in enumerate(rows, 1):
        sin_theta = user_lambda / (2 * d)
        if abs(sin_theta) <= 1.0:
            two_theta = f'{2 * np.degrees(np.arcsin(sin_theta)):>10.4f}'
        else:
            two_theta = f'{"n/a":>10}'
        print(f'{i:>3}  {hkl_str:<28}  {d:>10.4f}  {q:>10.4f}  {two_theta}')
    if args.min_d is not None:
        print(f'\nTotal: {len(rows)} reflections (d >= {args.min_d:.2f} Angstrom)')
    else:
        print(f'\nListed first {len(rows)} reflections')


if __name__ == '__main__':
    main()
