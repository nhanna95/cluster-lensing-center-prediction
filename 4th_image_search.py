from pathlib import Path
import numpy as np

import util.data as du
import util.graphing as gu
import util.math as mu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTER = du.clusters.get('clj2325')

KNOWN_TRIPLET = np.array([
    [2952.0226, 3439.9377],
    [3234.1333, 3278.7321],
    [4002.0593, 3615.1041],
], dtype=float)

BOX_FILTER = True
WEIGHT_INTERSECTIONS = True

PLOT_SYSTEM_HYPERBOLAS = True
PLOT_SYSTEM_IMAGES = True
PLOT_SYSTEM_ASYMPTOTES = False
PLOT_INTERSECTIONS = True
PLOT_TRIPLET_HYPERBOLA = True
PLOT_TRIPLET_ELLIPSE = True
PLOT_TRIPLET_ASYMPTOTES = True
PLOT_TRIPLET_POINTS = True
PLOT_PREDICTED_CENTER = True

SAVE_PLOT = False
SHOW_PLOT = True
LINE_WIDTH = None if SAVE_PLOT else 3


def _build_suffix():
    flags = []
    if PLOT_SYSTEM_HYPERBOLAS:
        flags.append('system_hyperbolas')
    if PLOT_SYSTEM_IMAGES:
        flags.append('system_images')
    if PLOT_SYSTEM_ASYMPTOTES:
        flags.append('system_asymptotes')
    if PLOT_TRIPLET_HYPERBOLA:
        flags.append('triplet_hyperbola')
    if PLOT_TRIPLET_ELLIPSE:
        flags.append('triplet_ellipse')
    if PLOT_TRIPLET_ASYMPTOTES:
        flags.append('triplet_asymptotes')
    if PLOT_TRIPLET_POINTS:
        flags.append('triplet_points')
    if PLOT_INTERSECTIONS:
        flags.append('intersections')
    if PLOT_PREDICTED_CENTER:
        flags.append('predicted_center')
    return f"_{'_'.join(flags)}" if flags else ''


FOLDER_PATH = Path(f'figures/{CLUSTER.folder_path}')
GRAPH_FILE = FOLDER_PATH / f'{CLUSTER.dataset}_triplet{_build_suffix()}.png'

FOLDER_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_predicted_center(cluster, box_filter=True, weight_inters=True):
    """Replicate the legacy multi-hyperbola intersection centre estimate."""
    systems = cluster.systems
    centre_offset = cluster.center_offset

    hyper_coeffs = []
    asymptotes = []
    side_flags = []
    fit_errors = []

    x_min = y_min = np.inf
    x_max = y_max = -np.inf

    for system in systems:
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
            if weight_inters:
                denom = max(fit_errors[i] * fit_errors[j], 1e-12)
                weight /= denom
            inter_wts.extend([weight] * pts.shape[0])

    if not inter_pts:
        raise RuntimeError('No intersections found; cannot estimate centre.')

    inter_pts = np.vstack(inter_pts)
    inter_wts = np.asarray(inter_wts, dtype=float)

    if box_filter:
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
    predicted = np.average(inter_pts, axis=0, weights=inter_wts)

    return predicted, {
        'intersections': inter_pts,
        'weights': inter_wts,
        'hyperbolas': hyper_coeffs,
        'asymptotes': asymptotes,
        'side_flags': side_flags,
        'fit_errors': np.asarray(fit_errors),
        'bounds': (x_min, x_max, y_min, y_max),
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main():
    cluster = CLUSTER
    predicted_rel, diagnostics = _compute_predicted_center(
        cluster,
        box_filter=BOX_FILTER,
        weight_inters=WEIGHT_INTERSECTIONS,
    )
    predicted_abs = predicted_rel + cluster.center_offset
    systems = cluster.systems
    
    print('--- Predicted Source Centre ---')
    print('Relative (pixels):', np.round(predicted_rel, 4))
    print('Absolute (pixels):', np.round(predicted_abs, 4))
    print()

    triplet_rel = KNOWN_TRIPLET - cluster.center_offset
    hyper_points = np.vstack((triplet_rel, predicted_rel))

    triplet_hyper = mu.generate_hyperbola(hyper_points)
    hyper_center = np.array(mu.find_conic_center(triplet_hyper))
    asym1, asym2 = mu.get_asymptotes(triplet_hyper)
    
    print('--- Triplet Hyperbola ---')
    print('Coefficients (A, B, C, D, E, F):', np.round(triplet_hyper, 6))
    print('Centre (relative pixels):', np.round(hyper_center, 6))
    print('Asymptotes: m1={:.6f}, b1={:.6f}; m2={:.6f}, b2={:.6f}'.format(
        asym1[0], asym1[1], asym2[0], asym2[1]
    ))
    print()

    if not np.allclose(hyper_center, predicted_rel, atol=1e-6):
        print('Warning: hyperbola centre does not match predicted centre exactly.')

    ellipse_meta = mu.ellipse_from_triplet(
        triplet_rel,
        triplet_hyper,
        center_point=predicted_rel,
        return_meta=True,
    )
    ellipse_coeffs = ellipse_meta['coeffs']
    ellipse_center = mu.find_conic_center(ellipse_coeffs)

    primary_axis = ellipse_meta['rotation'][:, 0]
    axis_angle = np.degrees(np.arctan2(primary_axis[1], primary_axis[0]))

    print('--- Constrained Ellipse ---')
    print('Coefficients (A, B, C, D, E, F):', np.round(ellipse_coeffs, 6))
    print('Centre (relative pixels):', np.round(ellipse_center, 6))
    print('Semi-axes (aligned with asymptotes): a={:.6f}, b={:.6f}'.format(
        ellipse_meta['axes'][0], ellipse_meta['axes'][1]
    ))
    print('Primary axis angle (degrees): {:.3f}'.format(axis_angle))
    print('Max residual on input points:', ellipse_meta['residual'])
    if ellipse_meta.get('approximate', False):
        print('Warning: ellipse fit required a small relaxation.')
    print()

    print('--- Diagnostics ---')
    print('Intersection count:', diagnostics['intersections'].shape[0])
    print('Mean intersection weight:', diagnostics['weights'].mean())
    print('Fit errors per quad (pixels):', np.round(diagnostics['fit_errors'], 6))

    base_points = np.vstack((cluster.all_points, KNOWN_TRIPLET, predicted_abs))
    graph_limits = mu.get_graph_limits(base_points - cluster.center_offset)
    gx_min, gx_max = graph_limits[:, 0]

    graph = gu.Graph(
        title=f'{cluster.name} Triplet Geometry',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)',
    )
    graph.update_axes(xaxis_range=graph_limits[:, 0],
                      yaxis_range=graph_limits[:, 1])

    if PLOT_SYSTEM_HYPERBOLAS or PLOT_SYSTEM_IMAGES or PLOT_SYSTEM_ASYMPTOTES:
        for idx, system in enumerate(systems):
            images_rel = system.images.astype(float) - cluster.center_offset
            hyper = diagnostics['hyperbolas'][idx]
            (m1, b1), (m2, b2) = diagnostics['asymptotes'][idx]
            flags = diagnostics['side_flags'][idx]

            if PLOT_SYSTEM_HYPERBOLAS:
                xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(
                    hyper, 2000, gx_min, gx_max
                )
                mask1 = mu.create_side_flag_mask(
                    xs1, ys1, [m1, m2], [b1, b2], flags
                )
                mask2 = mu.create_side_flag_mask(
                    xs2, ys2, [m1, m2], [b1, b2], flags
                )
                graph.add_trace(
                    xs1[mask1], ys1[mask1], 'hyperbola',
                    color_idx=idx, line_width=LINE_WIDTH,
                    name=f'System {system.name} Hyperbola'
                )
                graph.add_trace(
                    xs2[mask2], ys2[mask2], 'hyperbola',
                    color_idx=idx, line_width=LINE_WIDTH,
                    showlegend=False
                )

            if PLOT_SYSTEM_ASYMPTOTES:
                ax1, ay1 = mu.generate_linear_linspaces(m1, b1, 1000, gx_min, gx_max)
                ax2, ay2 = mu.generate_linear_linspaces(m2, b2, 1000, gx_min, gx_max)
                graph.add_trace(
                    ax1, ay1, 'asymptote', color_idx=idx,
                    line_width=LINE_WIDTH, showlegend=False
                )
                graph.add_trace(
                    ax2, ay2, 'asymptote', color_idx=idx,
                    line_width=LINE_WIDTH, showlegend=False
                )

            if PLOT_SYSTEM_IMAGES:
                graph.add_trace(
                    images_rel[:, 0], images_rel[:, 1], 'images',
                    color_idx=idx, showlegend=False,
                    name=f'System {system.name} Images'
                )

    if PLOT_TRIPLET_HYPERBOLA:
        tx1, ty1, tx2, ty2 = mu.generate_conic_linspaces(
            triplet_hyper, 2000, gx_min, gx_max
        )
        graph.add_trace(
            tx1, ty1, 'hyperbola', color_idx=len(systems),
            line_width=LINE_WIDTH, name='Triplet Hyperbola'
        )
        graph.add_trace(
            tx2, ty2, 'hyperbola', color_idx=len(systems),
            line_width=LINE_WIDTH, showlegend=False
        )

    if PLOT_TRIPLET_ASYMPTOTES:
        for idx_asym, (m_asym, b_asym) in enumerate((asym1, asym2)):
            ax, ay = mu.generate_linear_linspaces(m_asym, b_asym, 1000, gx_min, gx_max)
            graph.add_trace(
                ax, ay, 'asymptote', color_idx=len(systems),
                line_width=LINE_WIDTH,
                name='Triplet Asymptote' if idx_asym == 0 else None,
                showlegend=idx_asym == 0
            )

    if PLOT_TRIPLET_ELLIPSE:
        A, B, C, D, E, F = ellipse_coeffs
        ex, ey = mu.ellipse_linspace(A, B, C, D, E, F, 2000)
        graph.add_trace(
            ex, ey, 'ellipse', color_idx=len(systems) + 1,
            line_width=LINE_WIDTH, name='Triplet Ellipse'
        )

    if PLOT_TRIPLET_POINTS:
        graph.add_trace(
            triplet_rel[:, 0], triplet_rel[:, 1], 'images',
            color_idx=len(systems) + 1, name='Known Triplet',
            showlegend=True
        )

    if PLOT_INTERSECTIONS:
        graph.add_trace(
            diagnostics['intersections'][:, 0],
            diagnostics['intersections'][:, 1],
            'intersection', showlegend=False
        )

    if PLOT_PREDICTED_CENTER:
        graph.add_trace(
            predicted_rel[0], predicted_rel[1], 'predicted_center',
            name='Predicted Center'
        )

    graph.fig.update_layout(
        title=f'{cluster.name} Triplet Geometry (Predicted Center at {np.round(predicted_rel, 2)})'
    )

    if SAVE_PLOT:
        GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
        graph.save(GRAPH_FILE.as_posix())

    if SHOW_PLOT:
        graph.show()


if __name__ == '__main__':
    if not (SAVE_PLOT or SHOW_PLOT):
        raise SystemExit('Error: must either save or show the plot.')
    main()

