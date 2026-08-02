#!/usr/bin/env python3
"""Print diffraction peak list from space group and lattice parameters.

Uses xrayutilities to compute symmetry-allowed reflections.

Examples:
    python xrd_peaks.py --mat ceria
    python xrd_peaks.py -sg 225 -l 5.411
    python xrd_peaks.py -sg 225 -l 5.411 --energy 65.351 --qmax 8
    python xrd_peaks.py -sg 194 -l 2.95 4.68 --dmin 1.0
"""

import argparse
import sys
import numpy as np
import xrayutilities as xu

MATERIALS = {
    'ceria':  {'name': 'CeO2(SRM674b)',  'spacegroup': 225, 'lattice': [5.411526],
               'aliases': ['ceo2','ceria674b']},
    'lab6':   {'name': 'LaB6(SRM660c)',  'spacegroup': 221, 'lattice': [4.156826],
               'aliases': ['lab6c', 'lanthanumboride']},
    'lab6b':  {'name': 'LaB6(SRM660b)',  'spacegroup': 221, 'lattice': [4.156890],
               'aliases': ['']},
}

MAT_LOOKUP = {}
for _key, _mat in MATERIALS.items():
    MAT_LOOKUP[_key] = _mat
    for _alias in _mat.get('aliases', []):
        MAT_LOOKUP[_alias] = _mat


def parse_args():
    mat_names = ', '.join(MATERIALS.keys())
    p = argparse.ArgumentParser(description='List diffraction peaks for a given space group and lattice.')
    p.add_argument('--mat', type=str.lower, default=None,
                   help=f'Predefined material ({mat_names})')
    p.add_argument('-sg', '--spacegroup', type=int, default=None,
                   help='Space group number (1-230)')
    p.add_argument('-l', '--lattice', type=float, nargs='+', default=None,
                   help='Lattice parameters (1-6 floats depending on crystal system)')
    p.add_argument('-e', '--energy', type=float, default=71.676,
                   help='X-ray energy in keV (default: 71.676)')
    p.add_argument('--qmax', type=float, default=8.0,
                   help='Max Q in 1/Å (default: 8.0)')
    p.add_argument('--dmin', type=float, default=None,
                   help='Min d-spacing in Å (overrides --qmax)')
    p.add_argument('--sort', choices=['q', 'd', 'hkl'], default='q',
                   help='Sort order (default: q)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mat:
        if args.mat not in MAT_LOOKUP:
            print(f'Unknown material: {args.mat}')
            print(f'Available: {", ".join(MATERIALS.keys())}')
            sys.exit(1)
        mat = MAT_LOOKUP[args.mat]
        sg = args.spacegroup or mat['spacegroup']
        lat = args.lattice or mat['lattice']
        mat_name = mat['name']
    elif args.spacegroup and args.lattice:
        sg = args.spacegroup
        lat = args.lattice
        mat_name = None
    else:
        print('Provide either --mat or both -sg and -l')
        sys.exit(1)

    qmax = 2 * np.pi / args.dmin if args.dmin else args.qmax
    wavelength = 12.398419057638671 / args.energy

    lattice = xu.materials.SGLattice(sg, *lat)
    hkls = lattice.get_allowed_hkl(qmax)

    groups = {}
    for hkl in hkls:
        q = round(np.linalg.norm(lattice.GetQ(hkl)), 6)
        if q not in groups:
            groups[q] = []
        groups[q].append(hkl)

    results = []
    for q, hkl_list in groups.items():
        rep = sorted(hkl_list, key=lambda h: (-h[0], -h[1], -h[2]))[0]
        d = 2 * np.pi / q
        sin_theta = q * wavelength / (4 * np.pi)
        if sin_theta > 1:
            continue
        tth = 2 * np.degrees(np.arcsin(sin_theta))
        results.append((rep, d, q, tth, len(hkl_list)))

    if args.sort == 'q':
        results.sort(key=lambda x: x[2])
    elif args.sort == 'd':
        results.sort(key=lambda x: -x[1])
    else:
        results.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

    # Header
    sg_name = lattice.name
    lat_str = ', '.join(f'{v:.6f}' for v in lat)
    title = f'Material: {mat_name}    ' if mat_name else ''
    print(f'{title}Space group: {sg} ({sg_name})    Lattice: {lat_str} Å')
    print(f'Energy: {args.energy:.3f} keV  (λ={wavelength:.6f} Å)    '
          f'Q range: 0 – {qmax:.1f} 1/Å')
    print()
    print(f'{"No.":>4s}  {"hkl":>12s}  {"d (Å)":>8s}  {"Q (1/Å)":>8s}  {"2θ (deg)":>8s}  {"mult":>4s}')
    print('-' * 54)
    for i, (hkl, d, q, tth, mult) in enumerate(results, 1):
        print(f'  {i:2d}  ({hkl[0]:2d} {hkl[1]:2d} {hkl[2]:2d})  {d:8.4f}  {q:8.3f}  {tth:8.3f}  {mult:4d}')
    print(f'\nTotal: {len(results)} unique reflections')


if __name__ == '__main__':
    main()
