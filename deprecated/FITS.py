from pathlib import Path

import numpy as np
import fitsio
import matplotlib.pyplot as plt

import util.math as mu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOW_PLOT     = False      # preview the final mask
SAVE_INVERTED = True       # write the simple inverted image

PLOT_ELLIPSE  = True       # include ellipse contribution
PLOT_MINOR    = True       # minor hyperbola branch
PLOT_MAJOR    = True       # major hyperbola branch

FOLDER_PATH        = Path('fits/DESJ1134-21')
ORIG_DATA_FILE     = FOLDER_PATH / 'original_data' / 'DELJ113440.5-210322_i3.048.fits'

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_suffix():
    '''Return filename suffix encoding which curve layers are active.'''
    if PLOT_ELLIPSE and PLOT_MAJOR and PLOT_MINOR:
        return '_all'
    if PLOT_ELLIPSE and PLOT_MAJOR:
        return '_ellipse_major'
    if PLOT_ELLIPSE and PLOT_MINOR:
        return '_ellipse_minor'
    if PLOT_ELLIPSE:
        return '_ellipse'
    if PLOT_MAJOR and PLOT_MINOR:
        return '_hyperbola'
    if PLOT_MAJOR:
        return '_major'
    if PLOT_MINOR:
        return '_minor'
    return ''   # at least one flag is always true

def nanify(*arrays):
    '''Replace arrays with NaNs to keep shapes but hide data.'''
    return tuple(np.full_like(a, np.nan) for a in arrays)

def apply_branch_filters(xs1, ys1, xs2, ys2, images, asym1, asym2):
    '''
    Keep / drop hyperbola branches according to PLOT_MAJOR / PLOT_MINOR.
    '''
    m1, b1 = asym1
    m2, b2 = asym2

    if PLOT_MAJOR and PLOT_MINOR:
        return xs1, ys1, xs2, ys2
    if not (PLOT_MAJOR or PLOT_MINOR):
        return nanify(xs1, ys1, xs2, ys2)

    flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
    if PLOT_MINOR:
        flags = [not f for f in flags]

    mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], flags)
    mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], flags)
    return xs1[mask1], ys1[mask1], xs2[mask2], ys2[mask2]

def compute_darkening(hyp_pts, ell_pts, dshape, vmin, vmax):
    '''
    Build the darkening map from nearest-distance fields to the curve points.
    '''
    xs1, ys1, xs2, ys2 = hyp_pts
    ex1, ey1, ex2, ey2 = ell_pts

    # Convert NaNs to sentinel far-away coordinates
    arrays = xs1, ys1, xs2, ys2, ex1, ey1, ex2, ey2
    xs1, ys1, xs2, ys2, ex1, ey1, ex2, ey2 = [
        np.nan_to_num(a, nan=1e10) for a in arrays
    ]

    y_idx, x_idx = np.indices(dshape)

    # Min distance to either hyperbola branch
    dh1 = np.sqrt((x_idx - xs1[:, None, None])**2 + (y_idx - ys1[:, None, None])**2).min(axis=0)
    dh2 = np.sqrt((x_idx - xs2[:, None, None])**2 + (y_idx - ys2[:, None, None])**2).min(axis=0)
    dist_h = np.minimum(dh1, dh2)

    # Min distance to either ellipse branch
    de1 = np.sqrt((x_idx - ex1[:, None, None])**2 + (y_idx - ey1[:, None, None])**2).min(axis=0)
    de2 = np.sqrt((x_idx - ex2[:, None, None])**2 + (y_idx - ey2[:, None, None])**2).min(axis=0)
    dist_e = np.minimum(de1, de2)

    # Normalise distances then apply Gaussian fall-off
    half_width = 18.0
    h_norm  = 1.0 - np.clip(dist_h / half_width, 0.0, 1.0)
    e_norm  = 1.0 - (dist_e / dist_e.max() if dist_e.max() else dist_e)
    sigma   = 0.12
    dark_h  = 1e-11 * np.exp((h_norm**2) / (2.0 * sigma**2)) / sigma
    dark_e  = 1e-11 * np.exp((e_norm**2) / (2.0 * sigma**2)) / sigma
    return np.clip(dark_h + dark_e, vmin, vmax)

def plot_preview(mask, images, limits, x_min, x_max):
    '''Quick-look visualisation for debugging.'''
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.scatter(images[:, 0], images[:, 1], c='red', label='Image pts')
    plt.title('DES J1134-21 mask preview')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.xlim(x_min, x_max)
    plt.ylim(limits[:, 1])
    plt.legend()
    plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUFFIX = build_suffix()
IMG_OUT  = FOLDER_PATH / f'darkened_image{SUFFIX}.fits'
MASK_OUT = FOLDER_PATH / f'darkening{SUFFIX}.fits'
INV_OUT  = FOLDER_PATH / 'inverted.fits'

def main():
    '''End-to-end processing routine.'''
    data, hdr = fitsio.read(ORIG_DATA_FILE.as_posix(), header=True)
    dtype   = data.dtype
    h, w    = data.shape

    vmin, vmax = 0, np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    inverted   = vmax - data

    # Quad image centroids (1-based) → zero-based
    images = np.array([[17.96, 28.04],
                       [24.80, 25.16],
                       [27.92, 18.08],
                       [19.64, 20.60]], float) - 1.0

    limits = np.array([[0, 0], [w, h]], float)
    x_min, x_max = limits[:, 0]

    # Hyperbola
    hyp = mu.generate_hyperbola(images)
    (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
    xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(hyp, 2000, x_min, x_max)
    xs1, ys1, xs2, ys2 = apply_branch_filters(xs1, ys1, xs2, ys2, images, (m1, b1), (m2, b2))

    # Ellipse
    ell = mu.generate_optimal_ellipse(images, hyp)
    ex1, ey1, ex2, ey2 = mu.generate_conic_linspaces(ell, 2000, x_min, x_max)
    if not PLOT_ELLIPSE:
        ex1, ey1, ex2, ey2 = nanify(ex1, ey1, ex2, ey2)

    # Build mask and apply
    mask = compute_darkening((xs1, ys1, xs2, ys2), (ex1, ey1, ex2, ey2),
                              (h, w), vmin, vmax)
    dark_img = np.clip(inverted - mask, vmin, vmax)

    # Preview
    if SHOW_PLOT:
        plot_preview(mask, images, limits, x_min, x_max)

    # Write FITS
    fitsio.write(IMG_OUT.as_posix(), dark_img,  header=hdr, clobber=True)
    fitsio.write(MASK_OUT.as_posix(), (vmax - mask).astype(dtype),
                 header=hdr, clobber=True)
    if SAVE_INVERTED:
        fitsio.write(INV_OUT.as_posix(), inverted, header=hdr, clobber=True)


if __name__ == '__main__':
    print('Data shape :', fitsio.read(ORIG_DATA_FILE.as_posix()).shape)
    print('File suffix:', SUFFIX or '[none]')
    main()
