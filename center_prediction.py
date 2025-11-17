from pathlib import Path
import numpy as np

import util.data as du
import util.graphing as gu
import util.math as mu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Data selection ----------------------------------------------------------
CLUSTER = du.clusters.get('clj2325') # 'a1689' or 'clj2325'

DATASET = CLUSTER.dataset
SYSTEMS = CLUSTER.systems
CENTER_OFFSET = CLUSTER.center_offset

# Plot / output toggles -----------------------------------------------------
BOX_FILTER = True          # restrict intersections to quad bounding box
WEIGHT_INTERS = True      # weight intersections by distance error

PLOT_ELLIPSES = False
PLOT_ASYMS = True
PLOT_INTERS = True
PLOT_SOURCE = True
PLOT_IMAGES = True

SAVE_PLOT = False
SHOW_PLOT = True
LINE_WIDTH = None if SAVE_PLOT else 3  # thicker lines in live view

# TODO: add verbose settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _build_suffix():
    '''Return filename suffix encoding the active flag set.'''
    parts = []
    if PLOT_SOURCE:
        parts.append('source')
    if BOX_FILTER:
        parts.append('box_filter')
    if WEIGHT_INTERS:
        parts.append('weighted')
    if PLOT_ELLIPSES:
        parts.append('ellipses')
    if PLOT_ASYMS:
        parts.append('asymptotes')
    if PLOT_INTERS:
        parts.append('intersections')
    if PLOT_IMAGES:
        parts.append('images')
    return f"_{'_'.join(parts)}" if parts else ''

FOLDER_PATH = Path(f'figures/{CLUSTER.folder_path}')
GRAPH_FILE = FOLDER_PATH / f'{DATASET}{_build_suffix()}.png'

FOLDER_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_box(pts, x_min, x_max, y_min, y_max):
    '''Boolean mask selecting points inside the current bounding box.'''
    x, y = pts[:, 0], pts[:, 1]
    return (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    graph_limits = mu.get_graph_limits(CLUSTER.all_points - CENTER_OFFSET)
    gx_min, gx_max = graph_limits[:, 0]

    graph = gu.Graph(
        title=f'{CLUSTER.name} Quads',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)',
    )
    graph.update_axes(xaxis_range=graph_limits[:, 0],
                      yaxis_range=graph_limits[:, 1])

    # Book-keeping containers
    hyper_coeffs = []
    ellipse_coeffs = []
    asymptotes = []
    side_flags = []
    fit_errors = []

    # Dynamic bounding box for intersection filtering
    x_min = y_min = np.inf
    x_max = y_max = -np.inf

    # ---------------------------------------------------------------------
    # Per-system fits and layer plotting
    # ---------------------------------------------------------------------
    for idx, system in enumerate(SYSTEMS):
        name = system.name
        images = system.images.copy() - CENTER_OFFSET

        # Update bounding box
        x_min = min(x_min, images[:, 0].min())
        x_max = max(x_max, images[:, 0].max())
        y_min = min(y_min, images[:, 1].min())
        y_max = max(y_max, images[:, 1].max())

        images[images == 0] = 1e-10                   # avoid exact zeros

        # Hyperbola fit
        hyp = mu.generate_hyperbola(images)
        hyper_coeffs.append(hyp)

        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(
            hyp, 2000, gx_min, gx_max
        )

        # Keep only major branches
        flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
        side_flags.append(flags)
        asymptotes.append(((m1, b1), (m2, b2)))

        mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], flags)
        mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], flags)
        xs1, ys1 = xs1[mask1], ys1[mask1]
        xs2, ys2 = xs2[mask2], ys2[mask2]

        # Ellipse through quad (for optional plotting / error estimate)
        ell_coeffs = mu.ellipse_coefficients(images, hyp)
        ellipse_coeffs.append(ell_coeffs)
        ell_center = mu.find_conic_center(ell_coeffs)
        print(f'Quad {name} center (pixels): {ell_center[0] + CENTER_OFFSET[0]}, {ell_center[1] + CENTER_OFFSET[1]}')
        dist, nearest_point = mu.point_to_hyperbola_distance(
            ell_center[0], ell_center[1], hyp, return_nearest_point=True
        )
        print(f'Quad {name} nearest point hyperbola: {nearest_point[0] + CENTER_OFFSET[0]}, {nearest_point[1] + CENTER_OFFSET[1]}')
        fit_errors.append(
            dist
        )
        print(f'Quad {name} fit error: {fit_errors[-1]:.2f} pixels')

        # ---- Plot layers -------------------------------------------------
        graph.add_trace(xs1, ys1, 'hyperbola',
                        color_idx=idx, line_width=LINE_WIDTH,
                        name=f'Quad {name}')
        graph.add_trace(xs2, ys2, 'hyperbola',
                        color_idx=idx, line_width=LINE_WIDTH, showlegend=False)

        if PLOT_ELLIPSES:
            A, B, C, D, E, F = ell_coeffs
            ex, ey = mu.ellipse_linspace(A, B, C, D, E, F, 2000)
            graph.add_trace(ex, ey, 'ellipse', color_idx=idx,
                        line_width=LINE_WIDTH, name=f'Quad {name}', showlegend=True)

        if PLOT_SOURCE:
            graph.add_trace(ell_center[0], ell_center[1], 'source',
                            color_idx=idx, name=f'Predicted Center ({name})',
                            showlegend=False)

        if PLOT_ASYMS:
            ax1, ay1 = mu.generate_linear_linspaces(m1, b1, 2000, gx_min, gx_max)
            ax2, ay2 = mu.generate_linear_linspaces(m2, b2, 2000, gx_min, gx_max)
            graph.add_trace(ax1, ay1, 'asymptote', color_idx=idx, showlegend=False)
            graph.add_trace(ax2, ay2, 'asymptote', color_idx=idx, showlegend=False)

        if not SAVE_PLOT:
            graph.annotate_quads([(xs1, ys1), (xs2, ys2)], name)

        if PLOT_IMAGES:
            graph.add_trace(images[:, 0], images[:, 1], 'images',
                            color_idx=idx, name=f'Images ({name})',
                            showlegend=False)

    # ---------------------------------------------------------------------
    # Pairwise hyperbola intersections
    # ---------------------------------------------------------------------
    inter_pts, inter_wts = [], []
    
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

            pts = pts[mu.create_side_flag_mask(
                pts[:, 0], pts[:, 1], ms, bs, flags
            )]
            if pts.size == 0:
                continue
            
            inter_pts.append(pts)
            weight = 1.0
            if WEIGHT_INTERS:
                weight /= fit_errors[i] * fit_errors[j]
                # weight /= np.exp(fit_errors[i] + fit_errors[j])
            inter_wts.extend([weight] * pts.shape[0])

    if not inter_pts:
        raise RuntimeError('No intersections found; cannot estimate center.')

    inter_pts = np.vstack(inter_pts)
    inter_wts = np.asarray(inter_wts)

    # Bounding-box filter
    if BOX_FILTER:
        mask = _in_box(inter_pts, x_min, x_max, y_min, y_max)
        inter_pts, inter_wts = inter_pts[mask], inter_wts[mask]

    # Print RA/DEC of intersections
    for p in inter_pts:
        print('Intersection point:', p + CENTER_OFFSET)

    if PLOT_INTERS:
        graph.add_trace(inter_pts[:, 0], inter_pts[:, 1],
                        'intersection', showlegend=False)

    # Weight normalisation (sum → N)
    inter_wts *= inter_pts.shape[0] / inter_wts.sum()
    predicted = np.average(inter_pts, axis=0, weights=inter_wts)
    print('Predicted center (pixels):', predicted + CENTER_OFFSET)

    graph.add_trace(predicted[0], predicted[1],
                    'predicted_center', name='Predicted Center',
                    showlegend=False)
    graph.fig.update_layout(
        title=f'{CLUSTER.name} Quads (Predicted Center at {np.round(predicted, 2)})'
    )

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    if SAVE_PLOT:
        GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
        graph.save(GRAPH_FILE.as_posix())

    if SHOW_PLOT:
        graph.show()


if __name__ == '__main__':
    if not (SAVE_PLOT or SHOW_PLOT):
        raise SystemExit('Error: must either save or show the plot.')

    print('--------------------------------------------')
    print(f'Plotting: {CLUSTER}')
    print(f'Box filter: {BOX_FILTER}')
    print(f'Weight intersections: {WEIGHT_INTERS}')
    print(f'Plot ellipses: {PLOT_ELLIPSES}')
    print(f'Plot asymptotes: {PLOT_ASYMS}')
    print(f'Plot intersections: {PLOT_INTERS}')
    print(f'Plot source: {PLOT_SOURCE}')
    print(f'Plot images: {PLOT_IMAGES}')
    print(f'Show plot: {SHOW_PLOT}')
    print(f'Saving plot to file: {GRAPH_FILE.name}' if SAVE_PLOT else 'Save plot: False')
    print('--------------------------------------------')

    main()
