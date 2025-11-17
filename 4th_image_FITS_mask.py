from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import fitsio

import util.math as mu
import util.data as du

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTER = du.clusters.get('clj2325')
SYSTEMS = CLUSTER.systems

KNOWN_TRIPLET = np.array([
    [2952.0226, 3439.9377],
    [3234.1333, 3278.7321],
    [4002.0593, 3615.1041],
], dtype=float)

BOX_FILTER = True
WEIGHT_INTERSECTIONS = True

# Plot / output toggles -----------------------------------------------------
SHOW_PLOT = False
PLOT_ALL = False

PLOT_ELLIPSE = False
PLOT_MINOR = False
PLOT_MAJOR = False
PLOT_CIRCLES = False

# Triplet overlays ----------------------------------------------------------
PLOT_TRIPLET_HYPERBOLA = True
PLOT_TRIPLET_ELLIPSE = True
PLOT_TRIPLET_CIRCLES = True

# Gaussian fall-off parameters ---------------------------------------------
TARGET_HALF_WIDTH_PX = 8
RING_RADIUS_PX = 20
GAUSS_STD_PX = 0.35 if PLOT_ELLIPSE or PLOT_TRIPLET_ELLIPSE else 2

# I / O paths ---------------------------------------------------------------
FOLDER_PATH = Path(f'fits/{CLUSTER.folder_path}')
ORIGINAL_DATA_FILE = Path(CLUSTER.FITS_file)

FOLDER_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_suffix():
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
        parts.append('none')

    if PLOT_CIRCLES:
        parts.append('circles')
    if PLOT_TRIPLET_HYPERBOLA:
        parts.append('triplet_hyperbola')
    if PLOT_TRIPLET_ELLIPSE:
        parts.append('triplet_ellipse')
    if PLOT_TRIPLET_CIRCLES:
        parts.append('triplet_circles')
    return '_' + '_'.join(parts)


def _nanify(*arrays):
    return tuple(np.full_like(arr, np.nan) for arr in arrays)


def _apply_branch_filters(xs1, ys1, xs2, ys2, images, asym1, asym2):
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


def _compute_predicted_center():
    centre_offset = CLUSTER.center_offset

    hyper_coeffs = []
    asymptotes = []
    side_flags = []
    fit_errors = []

    x_min = y_min = np.inf
    x_max = y_max = -np.inf

    for system in SYSTEMS:
        images = system.images.astype(float) - centre_offset
        x_min = min(x_min, images[:, 0].min())
        x_max = max(x_max, images[:, 0].max())
        y_min = min(y_min, images[:, 1].min())
        y_max = max(y_max, images[:, 1].max())

        images[images == 0] = 1e-10

        hyp = mu.generate_hyperbola(images)
        hyper_coeffs.append(hyp)

        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        asymptotes.append(((m1, b1), (m2, b2)))
        flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
        side_flags.append(flags)

        ell_coeffs = mu.ellipse_coefficients(images, hyp)
        ell_center = mu.find_conic_center(ell_coeffs)
        dist = mu.point_to_hyperbola_distance(
            ell_center[0], ell_center[1], hyp
        )
        fit_errors.append(max(dist, 1e-12))

    inter_pts = []
    inter_wts = []

    for i in range(len(hyper_coeffs)):
        for j in range(i + 1, len(hyper_coeffs)):
            pts = mu.conic_intersections(hyper_coeffs[i], hyper_coeffs[j])
            if pts.size == 0:
                continue

            ms = (
                asymptotes[i][0][0], asymptotes[i][1][0],
                asymptotes[j][0][0], asymptotes[j][1][0],
            )
            bs = (
                asymptotes[i][0][1], asymptotes[i][1][1],
                asymptotes[j][0][1], asymptotes[j][1][1],
            )
            flags = (
                side_flags[i][0], side_flags[i][1],
                side_flags[j][0], side_flags[j][1],
            )

            mask = mu.create_side_flag_mask(
                pts[:, 0], pts[:, 1], ms, bs, flags
            )
            pts = pts[mask]
            if pts.size == 0:
                continue

            inter_pts.append(pts)
            weight = 1.0
            if WEIGHT_INTERSECTIONS:
                denom = max(fit_errors[i] * fit_errors[j], 1e-12)
                weight /= denom
            inter_wts.extend([weight] * pts.shape[0])

    if not inter_pts:
        raise RuntimeError('No intersections found; cannot estimate centre.')

    inter_pts = np.vstack(inter_pts)
    inter_wts = np.asarray(inter_wts, dtype=float)

    if BOX_FILTER:
        mask = (
            (x_min <= inter_pts[:, 0]) & (inter_pts[:, 0] <= x_max) &
            (y_min <= inter_pts[:, 1]) & (inter_pts[:, 1] <= y_max)
        )
        inter_pts = inter_pts[mask]
        inter_wts = inter_wts[mask]
        if inter_pts.size == 0:
            raise RuntimeError('All intersections filtered out by bounding box.')

    total_wt = inter_wts.sum()
    if total_wt <= 0:
        inter_wts = np.ones_like(inter_wts)
        total_wt = inter_wts.sum()

    inter_wts *= inter_pts.shape[0] / total_wt
    predicted_rel = np.average(inter_pts, axis=0, weights=inter_wts)
    predicted_abs = predicted_rel + centre_offset
    return predicted_rel, predicted_abs, {
        'hyperbolas': hyper_coeffs,
        'asymptotes': asymptotes,
        'side_flags': side_flags,
        'fit_errors': np.asarray(fit_errors),
        'intersections': inter_pts,
        'weights': inter_wts,
    }


def _compute_triplet_geometry(predicted_abs):
    triplet_abs = KNOWN_TRIPLET.astype(float)
    predicted_abs = np.asarray(predicted_abs, dtype=float)

    triplet_zero = triplet_abs - 1.0
    predicted_zero = predicted_abs - 1.0

    hyper_points = np.vstack((triplet_zero, predicted_zero))
    triplet_hyper = mu.generate_hyperbola(hyper_points)
    triplet_asym = mu.get_asymptotes(triplet_hyper)

    ellipse_meta = mu.ellipse_from_triplet(
        triplet_zero,
        triplet_hyper,
        center_point=predicted_zero,
        return_meta=True,
    )
    triplet_ellipse = ellipse_meta['coeffs']
    return {
        'triplet_zero': triplet_zero,
        'triplet_abs': triplet_abs,
        'predicted_zero': predicted_zero,
        'predicted_abs': predicted_abs,
        'hyperbola': triplet_hyper,
        'ellipse': triplet_ellipse,
        'asymptotes': triplet_asym,
        'ellipse_meta': ellipse_meta,
    }


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
        fx = a * x_idx ** 2 + b * x_idx * y_idx + c2 * y_idx ** 2 + d * x_idx + e * y_idx + f
        dfx = 2 * a * x_idx + b * y_idx + d
        dfy = b * x_idx + 2 * c2 * y_idx + e
        grad = np.hypot(dfx, dfy)
        grad[grad == 0] = 1e-12
        return np.abs(fx) / grad

    gauss = lambda d: np.exp(-(d ** 2) / (2.0 * GAUSS_STD_PX ** 2))

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

    ellipse_gauss = 0.0
    if ellipse_coeffs is not None:
        ellipse_gauss = gauss(_implicit_distance(ellipse_coeffs))

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
    data, hdr = fitsio.read(ORIGINAL_DATA_FILE.as_posix(), header=True)
    dtype = data.dtype
    h, w = data.shape
    min_val = 0
    max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

    data = np.where(np.isnan(data), 1, data)

    inverted = max_val - data
    net_img = inverted.copy()
    net_mask = np.zeros_like(data, float)

    x_min, x_max = 0.0, float(w)

    predicted_rel, predicted_abs, diagnostics = _compute_predicted_center()
    triplet = _compute_triplet_geometry(predicted_abs)

    for quad in SYSTEMS:
        images = quad.images.copy() - 1
        name = quad.name

        print(f'Processing {name}…')

        hyp = mu.generate_hyperbola(images)
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
        net_img = np.clip(net_img - dark, min_val, max_val)

        out_dir = FOLDER_PATH / 'masks' / 'individual'

        out_dir.mkdir(parents=True, exist_ok=True)
        fitsio.write(out_dir / f'{name}_darkened_image{SUFFIX}.fits',
                     dark_img, header=hdr, clobber=True)
        fitsio.write(out_dir / f'{name}_darkening{SUFFIX}.fits',
                     (max_val - dark).astype(dtype), header=hdr, clobber=True)

    triplet_zero = triplet['triplet_zero']
    triplet_hyper = triplet['hyperbola']
    triplet_ellipse = triplet['ellipse'] if PLOT_TRIPLET_ELLIPSE else None

    triplet_dark = _compute_darkening(
        (h, w),
        hyper_coeffs=triplet_hyper if PLOT_TRIPLET_HYPERBOLA else None,
        ellipse_coeffs=triplet_ellipse,
        asym1=triplet['asymptotes'][0],
        asym2=triplet['asymptotes'][1],
        images=None,
        circle_centers=triplet_zero if PLOT_TRIPLET_CIRCLES else None,
        min_value=min_val,
        max_value=max_val,
    )

    triplet_dark_img = np.clip(inverted - triplet_dark, min_val, max_val)
    net_mask = np.clip(net_mask + triplet_dark, min_val, max_val)
    net_img = np.clip(net_img - triplet_dark, min_val, max_val)

    hy_dir = FOLDER_PATH / 'masks' / 'triplet'
    hy_dir.mkdir(parents=True, exist_ok=True)
    fitsio.write(hy_dir / f'triplet_darkened_image{SUFFIX}.fits',
                 triplet_dark_img, header=hdr, clobber=True)
    fitsio.write(hy_dir / f'triplet_darkening{SUFFIX}.fits',
                 (max_val - triplet_dark).astype(dtype), header=hdr, clobber=True)

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


if __name__ == '__main__':
    print(f'Plot ellipse: {PLOT_ELLIPSE}')
    if PLOT_MAJOR and PLOT_MINOR:
        print('Plot hyperbola: True (both branches)')
    else:
        print(f'Plot major branch: {PLOT_MAJOR}')
        print(f'Plot minor branch: {PLOT_MINOR}')
    print(f'Plot circles: {PLOT_CIRCLES}')
    print(f'Plot triplet hyperbola: {PLOT_TRIPLET_HYPERBOLA}')
    print(f'Plot triplet ellipse: {PLOT_TRIPLET_ELLIPSE}')
    print(f'Plot triplet circles: {PLOT_TRIPLET_CIRCLES}')
    main()
