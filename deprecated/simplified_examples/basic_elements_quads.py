import numpy as np

import util.data as du
import util.graphing as gu
import util.math as mu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LINE_WIDTH       = None    # `None` → library default; set e.g. 3 for thicker
PLOT_MAJOR       = False   # if True, show only the major hyperbola branch
PLOT_SOURCE      = True    # draw predicted ellipse centre
PLOT_ASYMPTOTES  = True    # draw the two asymptotes

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    '''Generate a figure per system in *basic_elements_quads*.'''
    for system in du.basic_elements_quads:
        name   = system.name
        images = system.images.copy()

        print(f'Plotting {name} …')
        limits = mu.get_graph_limits(images)
        x_min, x_max = limits[:, 0]

        graph = gu.Graph(title=name, xaxis_title='X', yaxis_title='Y')
        graph.update_axes(xaxis_range=limits[:, 0], yaxis_range=limits[:, 1])

        # Hyperbola fit
        hyp = mu.generate_hyperbola(images)
        (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
        xs1, ys1, xs2, ys2 = mu.generate_conic_linspaces(hyp, 2000, x_min, x_max)

        # Keep major branch only if requested
        if PLOT_MAJOR:
            flags = mu.get_major_side_flag(images, m1, b1, m2, b2)
            mask1 = mu.create_side_flag_mask(xs1, ys1, [m1, m2], [b1, b2], flags)
            mask2 = mu.create_side_flag_mask(xs2, ys2, [m1, m2], [b1, b2], flags)
            xs1, ys1 = xs1[mask1], ys1[mask1]
            xs2, ys2 = xs2[mask2], ys2[mask2]

        if not ys1.size:
            print('Graphing error: no curve points generated; skipping.')
            continue

        # Ellipse fit
        ell     = mu.generate_optimal_ellipse(images, hyp)
        centre  = mu.find_conic_center(ell)
        ex1, ey1, ex2, ey2 = mu.generate_conic_linspaces(ell, 2000, x_min, x_max)

        # --- Plot layers --------------------------------------------------
        graph.add_trace(ex1, ey1, 'ellipse', line_width=LINE_WIDTH, showlegend=False)
        graph.add_trace(ex2, ey2, 'ellipse', line_width=LINE_WIDTH, showlegend=False)

        if PLOT_SOURCE:
            graph.add_trace(centre[0], centre[1], 'source',
                            name=f'Predicted ({name})', showlegend=False)

        if PLOT_ASYMPTOTES:
            ax1, ay1 = mu.generate_linear_linspaces(m1, b1, 2000, x_min, x_max)
            ax2, ay2 = mu.generate_linear_linspaces(m2, b2, 2000, x_min, x_max)
            graph.add_trace(ax1, ay1, 'asymptote', showlegend=False)
            graph.add_trace(ax2, ay2, 'asymptote', showlegend=False)

        graph.add_trace(xs1, ys1, 'hyperbola', line_width=LINE_WIDTH,
                        name=f'Quad {name}')
        graph.add_trace(xs2, ys2, 'hyperbola', line_width=LINE_WIDTH, showlegend=False)

        graph.add_trace(images[:, 0], images[:, 1], 'images',
                        name=f'Images ({name})', showlegend=False)

        graph.fig.update_layout(showlegend=False)
        graph.save(f'plots/basic_elements/{name}.png')


if __name__ == '__main__':
    main()
