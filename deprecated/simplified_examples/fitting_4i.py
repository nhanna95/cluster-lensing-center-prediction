import util.data as du
import util.graphing as gu
import util.math as mu

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    '''Generate and save a plot for every quad system.'''
    for system in du.original_systems:
        name   = system.name
        lens   = system.lens.copy()
        images = system.images.copy()

        # Avoid singularities at the origin
        images[images == 0] = 1e-10

        print(f'Plotting {name} …')

        graph_limits = mu.get_graph_limits(images)
        x_min, x_max = graph_limits[:, 0]

        coeffs = mu.generate_hyperbola(images)
        xs, ys1, ys2 = mu.generate_conic_linspaces(coeffs, 2000, x_min, x_max)

        if not ys1.size:
            print('Graphing error: no curve points generated.')
            continue

        graph = gu.Graph(title=name, xaxis_title='RA', yaxis_title='Dec')
        graph.update_axes(xaxis_range=graph_limits[:, 0],
                          yaxis_range=graph_limits[:, 1])

        # Hyperbola branches
        graph.add_trace(xs, ys1, 'hyperbola')
        graph.add_trace(xs, ys2, 'hyperbola', showlegend=False)

        # Observed data points
        graph.add_trace(images[:, 0], images[:, 1], 'images')
        graph.add_trace(lens[0, 0], lens[0, 1], 'lens')

        graph.save(f'plots/4i/{name}.png')


if __name__ == '__main__':
    main()
