import numpy as np

import util.data as du
import util.math as mu
import util.graphing as gu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_ALL     = True       # show all systems on one canvas
PLOT_HYPER   = True       # plot hyperbola branches
PLOT_MAJOR   = False      # restrict hyperbola to major branch
PLOT_SOURCE  = True       # mark predicted source position

SAVE_PLOT    = False      # save PNGs to disk
SHOW_PLOT    = True       # open an interactive window
LINE_WIDTH = None if SAVE_PLOT else 3  # thicker lines in live view

CLUSTER = du.clusters.get('clj2325') # 'a1689' or 'clj2325'

DATASET = CLUSTER.dataset
SYSTEMS = CLUSTER.systems
CENTER_OFFSET = CLUSTER.center_offset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    graph_limits = mu.get_graph_limits(CLUSTER.all_points - CENTER_OFFSET)
    gx_min, gx_max = graph_limits[:, 0]

    # If plotting all on one figure, create it up-front
    if PLOT_ALL:
        graph = gu.Graph(
            title=f'{CLUSTER.name} Quads',
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
        )
        graph.update_axes(xaxis_range=graph_limits[:, 0],
                        yaxis_range=graph_limits[:, 1])

    for idx, sys in enumerate(SYSTEMS):
        name  = sys.name[:-5]
        
        print(f'Processing {name} ({idx + 1}/{len(SYSTEMS)})')
        images = sys.images.copy() - CENTER_OFFSET

        # Per-system figure when not combining
        if not PLOT_ALL:
            graph = gu.Graph(name, 'X (pixels)', 'Y (pixels)')
            graph.update_axes(graph_limits[:, 0], graph_limits[:, 1])

        images[images == 0] = 1e-10                   # avoid singularity

        hyp = mu.generate_hyperbola(images)
        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(
            hyp, 2000, gx_min, gx_max
        )
        theta1, theta2 = np.arctan(m1), np.arctan(m2)

        # Keep only major branches if requested
        if PLOT_MAJOR:
            flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
            mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], flags)
            mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], flags)
            xs1, ys1 = xs1[mask1], ys1[mask1]
            xs2, ys2 = xs2[mask2], ys2[mask2]

        # Ellipse fit through quad images
        ell = mu.ellipse_coefficients(images, hyp)
        axes_ratio = mu.ellipse_axis_ratio(ell)
        print(f'Quad {name} axes ratio: {axes_ratio:.2f}')
        print(f'Quad {name} angles: {np.degrees(theta1):.2f}, {np.degrees(theta2):.2f}')
        A, B, C, D, E, F = ell
        ex, ey = mu.ellipse_linspace(A, B, C, D, E, F, 2000)
        center = mu.find_conic_center(ell)

        # ---- Plot layers -------------------------------------------------
        graph.add_trace(ex, ey, 'ellipse', color_idx=idx,
                        line_width=LINE_WIDTH, name=f'Quad {name}', showlegend=True)

        if PLOT_SOURCE:
            graph.add_trace(center[0], center[1], 'source',
                            color_idx=idx, name=f'Predicted ({name})',
                            showlegend=False)

        if PLOT_HYPER:
            graph.add_trace(xs1, ys1, 'hyperbola', color_idx=idx,
                            line_width=LINE_WIDTH, name=f'Quad {name}')
            graph.add_trace(xs2, ys2, 'hyperbola', color_idx=idx,
                            line_width=LINE_WIDTH, showlegend=False)

        # Annotate ends of branches for clarity (only when not saving)
        if not SAVE_PLOT:
            graph.annotate_quads([(xs1, ys1), (xs2, ys2)], name)

        # Observed image points
        graph.add_trace(images[:, 0], images[:, 1], 'images',
                        color_idx=idx, name=f'Images ({name})',
                        showlegend=False)

        # Save per-system figure if created
        if not PLOT_ALL and SAVE_PLOT:
            fname = f'{name}_ellipse'
            if PLOT_SOURCE: fname += '_source'
            if PLOT_MAJOR:  fname += '_major'
            out = f'plots/a1689/ellipses/{fname}.png'
            graph.save(out)
            if SHOW_PLOT:
                graph.show()

    # ---------------------------------------------------------------------
    # Combined output (if PLOT_ALL)
    # ---------------------------------------------------------------------
    if PLOT_ALL:
        title = f'ellipses{"_source" if PLOT_SOURCE else ""}'
        graph.fig.update_layout(title=f'Quad 4 vs Quad 42')
        graph.fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), title=None)

        if SAVE_PLOT:
            out = f'plots/a1689/ellipses/{title}.png'
            graph.save(out)

        if SHOW_PLOT:
            graph.show()

# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not (SAVE_PLOT or SHOW_PLOT):
        raise SystemExit('Error: must either save or show the plot.')

    print('--------------------------------------------')
    print('Plotting all systems' if PLOT_ALL else 'Plotting systems individually')
    print('Showing source points' if PLOT_SOURCE else 'Not showing source points')
    print('Saving plot to file' if SAVE_PLOT else 'Not saving plot to file')
    print('Showing plot' if SHOW_PLOT else 'Saving only')
    print('--------------------------------------------')

    main()
