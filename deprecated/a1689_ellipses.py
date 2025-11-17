import numpy as np

import util.data as du
import util.math as mu
import util.graphing as gu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_ALL     = True       # show all systems on one canvas
PLOT_MAJOR   = False       # restrict hyperbola to major branch
PLOT_SOURCE  = True       # mark predicted source position
SAVE_PLOT    = False      # save PNGs to disk
SHOW_PLOT    = True       # open an interactive window

systems = du.a1689_FITS_quads

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    line_w  = None if SAVE_PLOT else 5

    graph_limits = np.array([[-2500, -1300], [2500, 1600]])
    gx_min, gx_max = graph_limits[:, 0]

    # If plotting all on one figure, create it up-front
    if PLOT_ALL:
        graph = gu.Graph('Abell 1689 Quads', 'X (pixels)', 'Y (pixels)')
        graph.update_axes(graph_limits[:, 0], graph_limits[:, 1])

    for idx, sys in enumerate(systems):
        name  = sys.name[:-5]
        if name == '19':
            continue
        
        # idx = 1 if name == '42' else 2
        print(f'Processing {name} ({idx + 1}/{len(systems)})')
        images = sys.images.copy() - 2048.0           # recenter

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
        ex1, ey1, ex2, ey2 = mu.generate_conic_linspaces(ell, 2000, gx_min, gx_max)
        A, B, C, D, E, F = ell
        ex, ey = mu.ellipse_linspace(A, B, C, D, E, F, 2000)
        centre = mu.find_conic_center(ell)

        # ---- Plot layers -------------------------------------------------
        graph.add_trace(ex, ey, 'ellipse', color_idx=idx,
                        line_width=line_w, name=f'Quad {name}', showlegend=True)
        # graph.add_trace(ex1, ey1, 'ellipse', color_idx=idx,
        #                 line_width=line_w, name=f'Quad {name}', showlegend=True)
        # graph.add_trace(ex2, ey2, 'ellipse', color_idx=idx,
        #                 line_width=line_w, showlegend=False)

        if PLOT_SOURCE:
            graph.add_trace(centre[0], centre[1], 'source',
                            color_idx=idx, name=f'Predicted ({name})',
                            showlegend=False)

        # graph.add_trace(xs1, ys1, 'hyperbola', color_idx=idx,
        #                 line_width=line_w, name=f'Quad {name}')
        # graph.add_trace(xs2, ys2, 'hyperbola', color_idx=idx,
        #                 line_width=line_w, showlegend=False)

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
