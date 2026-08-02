"""Configuration parsing and ROI grouping for WAXS peak fitting."""

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def expand_peaks(config):
    """Expand peaks config into individual peak records with ROI intervals."""
    peaks = []
    for entry in config['peaks']:
        phase = entry['phase']
        for i, (c, r) in enumerate(zip(entry['center'], entry['roi'])):
            peaks.append({
                'phase': phase,
                'peak_idx': i,
                'center': float(c),
                'roi_min': float(c - r),
                'roi_max': float(c + r),
            })
    return peaks


def group_peaks(peaks):
    """Merge overlapping ROIs into fit groups (across all phases)."""
    sorted_peaks = sorted(peaks, key=lambda p: p['roi_min'])
    groups = []
    grp = [sorted_peaks[0]]
    grp_max = sorted_peaks[0]['roi_max']

    for pk in sorted_peaks[1:]:
        if pk['center'] <= grp_max:
            grp.append(pk)
            grp_max = max(grp_max, pk['roi_max'])
        else:
            groups.append(_make_group(grp, grp_max))
            grp = [pk]
            grp_max = pk['roi_max']
    groups.append(_make_group(grp, grp_max))
    return groups


def _make_group(peaks, roi_max):
    ordered = sorted(peaks, key=lambda p: p['center'])
    return {
        'peaks': ordered,
        'roi_min': min(p['roi_min'] for p in peaks),
        'roi_max': roi_max,
        'n_peaks': len(peaks),
    }
