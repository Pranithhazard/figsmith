"""
Helper utility functions for Figsmith
"""

import numpy as np


def to_float_array(values):
    """Convert array-like to float ndarray, coercing invalid entries to NaN."""
    arr = np.asarray(values)
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr)
    try:
        return arr.astype(float)
    except (ValueError, TypeError):
        flat = arr.ravel()
        converted = np.empty(flat.shape, dtype=float)
        for i, val in enumerate(flat):
            try:
                converted[i] = float(val)
            except (ValueError, TypeError):
                converted[i] = np.nan
        return converted.reshape(arr.shape)


def filter_valid_vectors(x, y, u, v, *extra_arrays):
    """Return arrays filtered to entries where base vectors are finite, along with mask."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    filtered = [arr[mask] for arr in (x, y, u, v)]
    extras = []
    for arr in extra_arrays:
        if arr is None:
            extras.append(None)
        else:
            extras.append(arr[mask])
    return (*filtered, *extras, mask)


def try_reshape_grid(x, y, *arrays):
    """Attempt to reshape flattened arrays to a regular grid. Returns (X, Y, reshaped_list) or None."""
    try:
        x_unique = np.unique(x)
        y_unique = np.unique(y)
    except Exception:
        return None

    nx = x_unique.size
    ny = y_unique.size

    if nx * ny != x.size or nx < 2 or ny < 2:
        return None

    # Sort by y, then x so reshape preserves grid structure
    order = np.lexsort((x, y))
    x_sorted = x[order]
    y_sorted = y[order]
    arrays_sorted = []
    for arr in arrays:
        if arr is None:
            arrays_sorted.append(None)
        else:
            arrays_sorted.append(arr[order])

    try:
        X = x_sorted.reshape(ny, nx)
        Y = y_sorted.reshape(ny, nx)
        reshaped = []
        for arr in arrays_sorted:
            if arr is None:
                reshaped.append(None)
            else:
                reshaped.append(arr.reshape(ny, nx))
    except Exception:
        return None

    return X, Y, reshaped


# Colormap options for dropdowns
CMAP_OPTIONS = [
    ('─── Sequential ───', 'viridis'),
    ('Greys', 'Greys'),
    ('Greys (reversed)', 'Greys_r'),
    ('Reds', 'Reds'),
    ('Reds (reversed)', 'Reds_r'),
    ('Blues', 'Blues'),
    ('Blues (reversed)', 'Blues_r'),
    ('Greens', 'Greens'),
    ('Greens (reversed)', 'Greens_r'),
    ('Viridis', 'viridis'),
    ('Plasma', 'plasma'),
    ('Inferno', 'inferno'),
    ('Magma', 'magma'),
    ('Cividis', 'cividis'),
    ('─── Diverging ───', 'RdBu_r'),
    ('Red-Blue', 'RdBu'),
    ('Red-Blue (reversed)', 'RdBu_r'),
    ('Red-Yellow-Blue', 'RdYlBu'),
    ('Cool-Warm', 'coolwarm'),
    ('Seismic', 'seismic'),
    ('─── Miscellaneous ───', 'jet'),
    ('Jet', 'jet'),
    ('Rainbow', 'rainbow'),
    ('Turbo', 'turbo'),
    ('Twilight', 'twilight')
]
