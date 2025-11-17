from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import fitsio

import util.math as mu
import util.data as du

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTER = du.clusters.get('clj2325') # 'a1689' or 'clj2325'
SYSTEMS = CLUSTER.systems

# Plot / output toggles -----------------------------------------------------
SHOW_PLOT   = False   # display the net inverted image at the end
PLOT_ALL    = False   # not used here but kept for parity with other scripts

PLOT_ELLIPSE = False  # include ellipse contribution
PLOT_MINOR   = False  # include minor hyperbola branch
PLOT_MAJOR   = True   # include major hyperbola branch
PLOT_CIRCLES = True  # include small Gaussian rings at image centres

# Gaussian fall-off parameters ---------------------------------------------
TARGET_HALF_WIDTH_PX = 8   # half-width in pixels used for normalisation
RING_RADIUS_PX       = 20   # radius (px) of optional darkening ring
GAUSS_STD_PX         = 0.35 if PLOT_ELLIPSE else 2 # σ in pixels

# I / O paths ---------------------------------------------------------------
FOLDER_PATH        = Path(f'fits/{CLUSTER.folder_path}')
ORIGINAL_DATA_FILE = Path(CLUSTER.FITS_file)

FOLDER_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_suffix():
    '''Return a filename suffix describing which plot layers are active.'''
    parts = []
    if PLOT_ELLIPSE and PLOT_MAJOR and PLOT_MINOR:
        parts.append('all')
    elif PLOT_ELLIPSE and PLOT_MAJOR:
        parts.append('ellipse_major')
    elif PLOT_ELLIPSE and PLOT_MINOR:
        parts.append('ellipse_minor')
    elif PLOT_ELLIPSE:
        parts.append('ellipse')
    elif PLOT_MAJOR and PLOT_MINOR:
        parts.append('hyperbola')
    elif PLOT_MAJOR:
        parts.append('major')
    elif PLOT_MINOR:
        parts.append('minor')
    else:
        raise ValueError('No plotting flags enabled – cannot build suffix.')

    if PLOT_CIRCLES:
        parts.append('circles')
    return '_' + '_'.join(parts)


def _nanify(*arrays):
    '''Return arrays filled with NaNs (shape preserved, values suppressed).'''
    return tuple(np.full_like(arr, np.nan) for arr in arrays)


def _apply_branch_filters(xs1, ys1, xs2, ys2, images, asym1, asym2):
    '''
    Keep or discard hyperbola branches according to PLOT_MAJOR / PLOT_MINOR.
    '''
    m1, b1 = asym1
    m2, b2 = asym2

    if PLOT_MAJOR and PLOT_MINOR:
        return xs1, ys1, xs2, ys2
    if not (PLOT_MAJOR or PLOT_MINOR):
        return _nanify(xs1, ys1, xs2, ys2)

    side_flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
    if PLOT_MINOR:
        side_flags = [not f for f in side_flags]

    mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], side_flags)
    mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], side_flags)

    return xs1[mask1], ys1[mask1], xs2[mask2], ys2[mask2]


def _compute_darkening(
    image_shape,
    hyper_coeffs=None,
    ellipse_coeffs=None,
    asym1=None,
    asym2=None,
    images=None,
    circle_centers=None,
    min_value=0.0,
    max_value=1.0,
):
    '''
    Build a floating-point mask (same shape as image) combining hyperbola,
    ellipse, and optional ring darkening contributions, each normalised and
    clipped to ``[min_value, max_value]``.
    '''
    h, w = image_shape
    if (hyper_coeffs is None and ellipse_coeffs is None and
            not circle_centers):
        return np.zeros((h, w), float)

    y_idx, x_idx = np.indices((h, w))

    def _normalise(arr):
        arr = np.clip(arr, 0, None)
        arr = (arr - arr.min()) / (np.ptp(arr) or 1.0)
        return arr * (max_value - min_value) + min_value

    def _implicit_distance(c):
        a, b, c2, d, e, f = c
        fx  = a*x_idx**2 + b*x_idx*y_idx + c2*y_idx**2 + d*x_idx + e*y_idx + f
        dfx = 2*a*x_idx + b*y_idx + d
        dfy = b*x_idx + 2*c2*y_idx + e
        grad = np.hypot(dfx, dfy)
        grad[grad == 0] = 1e-12
        return np.abs(fx) / grad

    gauss = lambda d: np.exp(-(d**2) / (2.0 * GAUSS_STD_PX**2))

    # -- Hyperbola ---------------------------------------------------------
    hyper_gauss = 0.0
    if hyper_coeffs is not None:
        dist = _implicit_distance(hyper_coeffs)

        if not (PLOT_MAJOR and PLOT_MINOR) and asym1 and asym2 and images is not None:
            flags = mu.get_major_side_flag(images, *asym1, *asym2)
            if not PLOT_MAJOR:
                flags = [not f for f in flags]
            m1, b1 = asym1
            m2, b2 = asym2
            sign1 = (y_idx - (m1 * x_idx + b1)) >= 0
            sign2 = (y_idx - (m2 * x_idx + b2)) >= 0
            mask = (sign1 == flags[0]) & (sign2 == flags[1])
            dist = np.where(mask, dist, TARGET_HALF_WIDTH_PX)

        hyper_gauss = gauss(dist)

    # -- Ellipse -----------------------------------------------------------
    ellipse_gauss = 0.0
    if ellipse_coeffs is not None:
        ellipse_gauss = gauss(_implicit_distance(ellipse_coeffs))

    # -- Ring(s) -----------------------------------------------------------
    ring_gauss = 0.0
    if circle_centers is not None:
        ring_dist = np.full((h, w), TARGET_HALF_WIDTH_PX, float)
        for cx, cy in circle_centers:
            r = np.hypot(x_idx - cx, y_idx - cy)
            ring_dist = np.minimum(ring_dist, np.abs(r - RING_RADIUS_PX))
        ring_gauss = gauss(ring_dist)

    total = _normalise(hyper_gauss) \
          + _normalise(ellipse_gauss) \
          + _normalise(ring_gauss)
    return np.clip(total, min_value, max_value)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUFFIX = _build_suffix()

def main():
    '''End-to-end processing routine.'''
    data, hdr = fitsio.read(ORIGINAL_DATA_FILE.as_posix(), header=True)
    dtype      = data.dtype
    h, w       = data.shape
    min_val    = 0
    max_val    = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    
    data = np.where(np.isnan(data), 1, data)
    
    inverted   = max_val - data
    net_img    = inverted.copy()
    net_mask   = np.zeros_like(data, float)

    x_min, x_max = 0.0, float(w)
    
    for quad in SYSTEMS:
        images = quad.images.copy() - 1  # 0-based
        name = quad.name
        
        print(f'Processing {name}…')

        hyp  = mu.generate_hyperbola(images)
        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(hyp, 2000, x_min, x_max)
        xs1, ys1, xs2, ys2 = _apply_branch_filters(
            xs1, ys1, xs2, ys2, images, (m1, b1), (m2, b2)
        )

        ell = mu.ellipse_coefficients(images, hyp)

        dark = _compute_darkening(
            (h, w),
            hyper_coeffs=hyp,
            ellipse_coeffs=ell if PLOT_ELLIPSE else None,
            asym1=(m1, b1), asym2=(m2, b2),
            images=images,
            circle_centers=images if PLOT_CIRCLES else None,
            min_value=min_val,
            max_value=max_val,
        )
        
        dark_img = np.clip(inverted - dark, min_val, max_val)
        net_mask = np.clip(net_mask + dark, min_val, max_val)
        net_img  = np.clip(net_img  - dark, min_val, max_val)
        
        out_dir = FOLDER_PATH / 'masks' / 'individual'
        
        out_dir.mkdir(parents=True, exist_ok=True)
        fitsio.write(out_dir / f'{name}_darkened_image{SUFFIX}.fits',
                     dark_img, header=hdr, clobber=True)
        fitsio.write(out_dir / f'{name}_darkening{SUFFIX}.fits',
                     (max_val - dark).astype(dtype), header=hdr, clobber=True)

    # Save combined outputs
    masks_dir = FOLDER_PATH / 'masks'
    masks_dir.mkdir(exist_ok=True)
    fitsio.write(masks_dir / f'net_image{SUFFIX}.fits',
                 net_img, header=hdr, clobber=True)
    fitsio.write(masks_dir / f'net_mask{SUFFIX}.fits',
                 net_mask, header=hdr, clobber=True)

    print(f'Processing complete.  Results saved to {FOLDER_PATH}.')
    if SHOW_PLOT:
        plt.imshow(net_img, cmap='gray', origin='lower')
        plt.title('Net Inverted Image')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.colorbar(label='Intensity')
        plt.show()

# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'Plot ellipse: {PLOT_ELLIPSE}')
    
    if PLOT_MAJOR and PLOT_MINOR:
        print('Plot hyperbola: True (both branches)')
    else:
        print(f'Plot major branch: {PLOT_MAJOR}')
        print(f'Plot minor branch: {PLOT_MINOR}')
        
    print(f'Plot circles: {PLOT_CIRCLES}')
    main()
