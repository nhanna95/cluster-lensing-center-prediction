import numpy as np

import util.data as du
import util.graphing as gu
import util.math as mu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_ALL      = False     # True → combine all systems on one figure
PLOT_MAJOR    = True      # keep only the major hyperbola branch
PLOT_SOURCE   = False     # mark rough source (image centroid)
SHOW_PLOT     = True      # display window; always saves PNG
BOX_FILTER    = True      # restrict intersections to quad envelope

DATASET       = 'all_quads'   # keys: all_quads / seven_sisters / three_miscreants …

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def rough_source(imgs):
    '''Return simple average of the four image positions.'''
    return np.mean(imgs, axis=0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    systems = du.quad_datasets.get(DATASET, du.a1689_quads)

    graph_limits = np.array([[-2000, -1300], [2000, 1600]])
    gx_min, gx_max = graph_limits[:, 0]

    if PLOT_ALL:
        graph = gu.Graph('Abell 1689 Quads', 'X (pixels)', 'Y (pixels)')
        graph.update_axes(graph_limits[:, 0], graph_limits[:, 1])

    # Containers for intersection pass
    hyper_coeffs, asymptotes, side_flags = [], [], []
    bb_xmin = bb_ymin =  np.inf
    bb_xmax = bb_ymax = -np.inf

    # ---------------------------------------------------------------------
    # Per-system plotting + bookkeeping
    # ---------------------------------------------------------------------
    for idx, sys in enumerate(systems):
        name   = sys.name
        images = sys.images.copy() - 2048.0   # recenter

        if not PLOT_ALL:
            graph = gu.Graph(name, 'X (pixels)', 'Y (pixels)')
            graph.update_axes(graph_limits[:, 0], graph_limits[:, 1])

        # Expand bounding-box for later BOX_FILTER
        bb_xmin = min(bb_xmin, images[:, 0].min())
        bb_xmax = max(bb_xmax, images[:, 0].max())
        bb_ymin = min(bb_ymin, images[:, 1].min())
        bb_ymax = max(bb_ymax, images[:, 1].max())

        images[images == 0] = 1e-10           # avoid origin singularity

        hyp = mu.generate_hyperbola(images)
        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(hyp, 2000, gx_min, gx_max)

        # Major/minor branch handling
        flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
        if PLOT_MAJOR:
            mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], flags)
            mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], flags)
            xs1, ys1 = xs1[mask1], ys1[mask1]
            xs2, ys2 = xs2[mask2], ys2[mask2]

        # Plot layers
        graph.add_trace(xs1, ys1, 'hyperbola', color_idx=idx, name=f'Quad {name}')
        graph.add_trace(xs2, ys2, 'hyperbola', color_idx=idx, showlegend=False)
        graph.add_trace(images[:, 0], images[:, 1], 'images',
                        color_idx=idx, name=f'Images ({name})', showlegend=False)

        if PLOT_SOURCE:
            src = rough_source(images)
            graph.add_trace(src[0], src[1], 'source',
                            color_idx=idx, name=f'Approx. source ({name})',
                            showlegend=False)

        if not PLOT_ALL:
            out = f'plots/a1689/hyperbolae/{name}{"_major" if PLOT_MAJOR else ""}{"_source" if PLOT_SOURCE else ""}.png'
            graph.save(out)

        # Book-keeping for global intersection estimate
        hyper_coeffs.append(hyp)
        asymptotes.append(((m1, b1), (m2, b2)))
        side_flags.append(flags)

    # ---------------------------------------------------------------------
    # Combined-figure processing
    # ---------------------------------------------------------------------
    if not PLOT_ALL:
        return

    inter_pts = []
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
            f  = (
                side_flags[i][0], side_flags[i][1],
                side_flags[j][0], side_flags[j][1],
            )
            mask = mu.create_side_flag_mask(pts[:, 0], pts[:, 1], ms, bs, f)
            pts  = pts[mask]
            if pts.size:
                inter_pts.append(pts)

    if not inter_pts:
        print('No intersections found.')
        return

    inter_pts = np.vstack(inter_pts)

    if BOX_FILTER:
        inter_pts = inter_pts[
            (bb_xmin <= inter_pts[:, 0]) & (inter_pts[:, 0] <= bb_xmax) &
            (bb_ymin <= inter_pts[:, 1]) & (inter_pts[:, 1] <= bb_ymax)
        ]

    graph.add_trace(inter_pts[:, 0], inter_pts[:, 1], 'intersection', showlegend=False)

    center = np.mean(inter_pts, axis=0)
    graph.add_trace(center[0], center[1], 'predicted_center',
                    name='Predicted Center', showlegend=False)

    title_suffix = (f'{"_source" if PLOT_SOURCE else ""}'
                    f'{"_box_filter" if BOX_FILTER else ""}')
    graph.fig.update_layout(
        title=f'Abell 1689 Quads - Predicted Center {np.round(center, 2)} ({DATASET}{title_suffix})'
    )

    out = f'plots/a1689/{DATASET}{title_suffix}.png'
    graph.save(out)
    if SHOW_PLOT:
        graph.show()


if __name__ == '__main__':
    print('Plotting all systems' if PLOT_ALL else 'Plotting systems individually')
    print('Plotting major branches' if PLOT_MAJOR else 'Plotting both branches')
    print('Showing source points' if PLOT_SOURCE else 'Not showing source points')
    print('Showing plot' if SHOW_PLOT else 'Saving plot only')
    print('Box filtering enabled' if BOX_FILTER else 'No box filtering')
    print('Dataset:', DATASET)
    print('--------------------------------------------')

    main()
