#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Script to print hdf5 content in tree mode. I only tested this on Linux.
#
# 01/12/24    Initial release
# 12/07/25    Added --value option to display field values
# 05/20/26    show value of sliced array. show attributes
#
# by AC (cchuang@anl.gov)
# $Revision: 1.2.0 $  $Date: 2026/05/20$
"""

import os
import sys
import argparse

# Number of elements to show at head and tail when not using --full
NUM_PREVIEW = 5


# check if input is a hdf5 file
def is_hdf5_file(filename):
    try:
        with h5py.File(filename, 'r'):
            pass
        return True
    except (OSError, ValueError):
        return False


# sub-function to construct tree view
def _print_attrs(item, pre):
    for akey, aval in item.attrs.items():
        val_str = str(aval)
        if len(val_str) > 80:
            val_str = val_str[:77] + '...'
        print(f'{pre}@ {akey} = {val_str}')


def h5_tree(val, pre='', show_attrs=False):
    if show_attrs and val.attrs:
        _print_attrs(val, pre)
    items = len(val)
    for key, item in val.items():
        items -= 1
        if items == 0:
            # the last item
            if isinstance(item, h5py._hl.group.Group):
                print(f'{pre}└── {key}')
                h5_tree(item, f'{pre}    ', show_attrs)
            else:
                shape_str = f' ({item.shape})' if hasattr(item, 'shape') else ''
                print(f'{pre}└── {key}{shape_str}')
                if show_attrs and item.attrs:
                    _print_attrs(item, f'{pre}    ')
        else:
            if isinstance(item, h5py._hl.group.Group):
                print(f'{pre}├── {key}')
                h5_tree(item, f'{pre}│   ', show_attrs)
            else:
                shape_str = f' ({item.shape})' if hasattr(item, 'shape') else ''
                print(f'{pre}├── {key}{shape_str}')
                if show_attrs and item.attrs:
                    _print_attrs(item, f'{pre}│   ')


def _parse_slice(spec):
    """Parse a single slice spec: '3' -> 3, '1:5' -> slice(1,5), ':' -> slice(None)."""
    spec = spec.strip()
    if ':' in spec:
        parts = spec.split(':')
        start = int(parts[0]) if parts[0].strip() else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        step = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
        return slice(start, stop, step)
    return int(spec)


def _parse_path_and_slice(field_path):
    """Split 'path[0,:,1:5]' into ('path', (0, slice(None), slice(1,5)))."""
    if '[' not in field_path:
        return field_path, None
    bracket = field_path.index('[')
    path = field_path[:bracket]
    slice_str = field_path[bracket + 1:].rstrip(']')
    slices = tuple(_parse_slice(s) for s in slice_str.split(','))
    return path, slices


def _print_1d(data, show_full):
    n = len(data)
    if show_full or n <= 2 * NUM_PREVIEW:
        print(f"Values ({n} elements):")
        for i, val in enumerate(data):
            print(f"  [{i}]: {val}")
    else:
        print(f"Values ({n} elements, showing first and last {NUM_PREVIEW}):")
        for i in range(NUM_PREVIEW):
            print(f"  [{i}]: {data[i]}")
        print(f"  ... ({n - 2 * NUM_PREVIEW} more elements) ...")
        for i in range(n - NUM_PREVIEW, n):
            print(f"  [{i}]: {data[i]}")


def _print_2d(data, show_full):
    import numpy as np
    nrows, ncols = data.shape
    print(f"Values ({nrows} x {ncols}):")

    if show_full or (nrows <= 2 * NUM_PREVIEW and ncols <= 2 * NUM_PREVIEW):
        row_indices = range(nrows)
        col_indices = range(ncols)
    else:
        row_indices = list(range(min(NUM_PREVIEW, nrows))) + \
                      list(range(max(nrows - NUM_PREVIEW, NUM_PREVIEW), nrows))
        col_indices = list(range(min(NUM_PREVIEW, ncols))) + \
                      list(range(max(ncols - NUM_PREVIEW, NUM_PREVIEW), ncols))

    # Format header
    col_strs = [f'{c:>12}' for c in col_indices]
    if not show_full and ncols > 2 * NUM_PREVIEW:
        col_strs.insert(NUM_PREVIEW, '         ...')
    print(f"{'':>8} " + ' '.join(col_strs))

    prev_row = -1
    for ri, r in enumerate(row_indices):
        if r - prev_row > 1 and prev_row >= 0:
            print(f"{'...':>8}")
        prev_row = r
        vals = [f'{data[r, c]:>12.6g}' for c in col_indices]
        if not show_full and ncols > 2 * NUM_PREVIEW:
            vals.insert(NUM_PREVIEW, '         ...')
        print(f"[{r:>5}]  " + ' '.join(vals))


def show_field_value(hf, field_path, show_full=False, show_attrs=False):
    """Display values of a dataset, with optional numpy-style slicing."""
    import numpy as np

    # Normalize path (ensure it starts with /)
    if not field_path.startswith('/'):
        field_path = '/' + field_path

    path, slices = _parse_path_and_slice(field_path)

    # Normalize dataset path
    if not path.startswith('/'):
        path = '/' + path

    # Check if the path exists
    if path not in hf:
        print(f"Error: Field '{path}' does not exist in the file.")
        print("\nAvailable paths:")
        def print_paths(group, prefix=''):
            for key in group.keys():
                full_path = f"{prefix}/{key}"
                print(f"  {full_path}")
                if isinstance(group[key], h5py._hl.group.Group):
                    print_paths(group[key], full_path)
        print_paths(hf)
        return False

    dataset = hf[path]

    if isinstance(dataset, h5py._hl.group.Group):
        if show_attrs and dataset.attrs:
            print(f"Group: {path}")
            print("-" * 40)
            for akey, aval in dataset.attrs.items():
                print(f"  {akey} = {aval}")
            return True
        print(f"Error: '{path}' is a group, not a dataset.")
        return False

    # Apply slice or read all
    if slices is not None:
        data = dataset[slices]
    else:
        data = dataset[()]

    # Convert bytes to str for string arrays
    if hasattr(data, 'dtype') and data.dtype.kind in ('S', 'O'):
        if isinstance(data, np.ndarray):
            data = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in data.flat]).reshape(data.shape)

    print(f"Field: {field_path}")
    print(f"Dataset shape: {dataset.shape}")
    if slices is not None:
        print(f"Sliced shape:  {data.shape if hasattr(data, 'shape') else '(scalar)'}")
    print(f"Dtype: {dataset.dtype}")
    print("-" * 40)

    if not hasattr(data, 'ndim') or data.ndim == 0:
        print(f"Value: {data}")
    elif data.ndim == 1:
        _print_1d(data, show_full)
    elif data.ndim == 2:
        _print_2d(data, show_full)
    else:
        print(f"Result is {data.ndim}D (shape: {data.shape}).")
        print("Use slicing to reduce to 1D or 2D, e.g.:")
        dims = ','.join(['0' if i == 0 else ':' for i in range(data.ndim)])
        print(f"  {os.path.basename(sys.argv[0])} file.h5 -v \"{path}[{dims}]\"")

    return True


if __name__ == '__main__':
    # check only non-standard library
    try:
        import h5py
    except ModuleNotFoundError:
        print("Warning: 'h5py' package is not installed. Please install it or switch to the proper conda environment.")
        exit(1)

    # Create a parser
    parser = argparse.ArgumentParser(
        description='A script to tree-view hdf5 content and inspect field values.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Examples:
  %(prog)s data.h5                              # Show tree structure
  %(prog)s data.h5 -v /group/field              # Show 1D field values
  %(prog)s data.h5 -v /group/field -f           # Show all values
  %(prog)s data.h5 -v "OmegaSumFrame[0,:,:]"    # Slice 3D -> 2D
  %(prog)s data.h5 -v "OmegaSumFrame[0,100,:]"  # Slice 3D -> 1D
  %(prog)s data.h5 -v "data[0:5,0:5]"           # Sub-region of 2D
  %(prog)s data.h5 -a                            # Show tree with attributes
  %(prog)s data.h5 -v /group -a                  # Show group attributes
        '''
    )

    # define required input
    parser.add_argument('filename', type=str, nargs='?', 
                        help='filename of the hdf5 file')
    
    # Add --value option
    parser.add_argument('--value', '-v', type=str, metavar='PATH',
                        help='path to a field to display its values (e.g., /group/dataset)')
    
    # Add --full option
    parser.add_argument('--full', '-f', action='store_true',
                        help='show all values instead of just first/last few')

    # Add --attrs option
    parser.add_argument('--attrs', '-a', action='store_true',
                        help='show attributes on groups and datasets')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    filename = args.filename

    if not filename:
        parser.print_help()
        exit()

    # Check if the file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        exit(1)

    if not is_hdf5_file(filename):
        print(f"Warning: '{os.path.basename(filename)}' is not an HDF5 file, you idiot!!")
        exit(1)

    with h5py.File(filename, 'r') as hf:
        if args.value:
            # Show field values
            show_field_value(hf, args.value, args.full, args.attrs)
        else:
            # Show tree structure (default behavior)
            print(f'{os.path.basename(filename)}')
            print('/')
            h5_tree(hf, show_attrs=args.attrs)

