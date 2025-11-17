from pathlib import Path

import numpy as np
import fitsio
from PIL import Image

import util.math as mu
import util.data as du

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Plot / output toggles -----------------------------------------------------
SHOW_PLOT   = True   # display the net inverted image at the end
PLOT_ALL    = True   # not used here but kept for parity with other scripts

# Gaussian fall-off parameters ---------------------------------------------
TARGET_HALF_WIDTH_PX = 8   # half-width in pixels used for normalisation
RING_RADIUS_PX       = 8   # radius (px) of optional darkening ring
GAUSS_STD_PX         = 1 # σ in pixels

# I / O paths ---------------------------------------------------------------
FOLDER_PATH        = Path('fits/A1689')
ORIGINAL_DATA_FILE = FOLDER_PATH / 'original_data' / \
                     'hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits'
                     
good_color = [0, 1, 1]
# bad_color = [1, 1, 1]
# bad_color = [0, 1, 0]
bad_color = [.6, .6, 0]

good_quads = ['4', '8', '9']

def _nanify(*arrays):
    '''Return arrays filled with NaNs (shape preserved, values suppressed).'''
    return tuple(np.full_like(arr, np.nan) for arr in arrays)

def _apply_branch_filters(xs1, ys1, xs2, ys2, images, asym1, asym2):
    m1, b1 = asym1
    m2, b2 = asym2

    side_flags = mu.get_major_side_flag(images, m1, b1, m2, b2)

    mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], side_flags)
    mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], side_flags)

    return xs1[mask1], ys1[mask1], xs2[mask2], ys2[mask2]

def _compute_darkening(
    image_shape,
    hyper_coeffs=None,
    asym1=None,
    asym2=None,
    images=None,
    min_value=0.0,
    max_value=1.0,
    good_quad=False
):
    '''
    Build a floating-point mask (same shape as image) combining hyperbola,
    ellipse, and optional ring darkening contributions, each normalised and
    clipped to ``[min_value, max_value]``. Applies red or magenta darkening
    based on ``good_quad`` flag.
    '''
    h, w = image_shape

    y_idx, x_idx = np.indices((h, w))

    def _normalise(arr):
        arr = np.clip(arr, 0, None)
        arr = (arr - arr.min()) / (arr.ptp() or 1.0)
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

        flags = mu.get_major_side_flag(images, *asym1, *asym2)
        m1, b1 = asym1
        m2, b2 = asym2
        sign1 = (y_idx - (m1 * x_idx + b1)) >= 0
        sign2 = (y_idx - (m2 * x_idx + b2)) >= 0
        mask = (sign1 == flags[0]) & (sign2 == flags[1])
        dist = np.where(mask, dist, TARGET_HALF_WIDTH_PX)

        hyper_gauss = gauss(dist)

    total = _normalise(hyper_gauss)

    # Create RGB mask
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
    if good_quad:
        rgb_mask[..., 0] = total * good_color[0]  # Red channel
        rgb_mask[..., 1] = total * good_color[1]  # Green channel
        rgb_mask[..., 2] = total * good_color[2]  # Blue channel
    else:
        rgb_mask[..., 0] = total * bad_color[0]  # Red channel
        rgb_mask[..., 1] = total * bad_color[1]  # Green channel
        rgb_mask[..., 2] = total * bad_color[2]  # Blue channel

    return np.clip(rgb_mask, min_value, max_value)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    '''End-to-end processing routine.'''
    data, hdr = fitsio.read(ORIGINAL_DATA_FILE.as_posix(), header=True)
    dtype      = data.dtype
    h, w       = data.shape
    min_val    = 0
    max_val    = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

    inverted   = max_val - data
    net_img    = inverted.copy()
    # Convert grayscale net_img to RGB
    rgb_img = np.zeros((h, w, 3), dtype=np.float32)
    rgb_img[..., 0] = net_img / max_val  # Red channel
    rgb_img[..., 1] = net_img / max_val  # Green channel
    rgb_img[..., 2] = net_img / max_val  # Blue channel

    x_min, x_max = 0.0, float(w)
    
    dataset = du.a1689_all_fits
    
    for quad in dataset:
        images = quad.images.copy() - 1  # 0-based
        name   = quad.name[:-5]          # strip ' FITS'

        print(f'Processing {name}…')

        hyp  = mu.generate_hyperbola(images)
        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(hyp, 2000, x_min, x_max)
        xs1, ys1, xs2, ys2 = _apply_branch_filters(
            xs1, ys1, xs2, ys2, images, (m1, b1), (m2, b2)
        )

        ell = mu.generate_optimal_ellipse(images, hyp)

        dark = _compute_darkening(
            (h, w),
            hyper_coeffs=hyp,
            asym1=(m1, b1), asym2=(m2, b2),
            images=images,
            min_value=min_val,
            max_value=max_val,
            good_quad=name in good_quads
        )

        rgb_img  = np.clip(rgb_img  - dark, min_val, max_val)


    # Flip the image vertically
    rgb_img = np.flipud(rgb_img)

    rgb_img_8bit = (rgb_img * 255).astype(np.uint8)
    
    #.35 x .35
    x_start_crop = int(w * 0.40)  # 5% from left
    x_end_crop = int(w * 0.55)    # 5% from right
    y_start_crop = int(h * 0.45)  # 5% from top
    y_end_crop = int(h * 0.60)    # 5% from bottom
    
    # Crop the image to remove 5% from each side
    rgb_img_8bit = rgb_img_8bit[y_start_crop:y_end_crop, x_start_crop:x_end_crop]
    
    # Save the RGB image as a JPEG file
    jpeg_path = 'teaser_fig.png'
    Image.fromarray(rgb_img_8bit).save(jpeg_path)

# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
