import numpy as np

import util.data as du
import util.math as mu
import util.graphing as gu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLOT_TRANSFORMED = False   # set True to use the XY frame
SAVE_DIR         = 'plots/3i_1s'   # output root

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(plot_transformed=False):
    '''Loop over quad systems and generate one plot per missing image.'''
    for system in du.original_systems:
        name   = system.name
        lens   = system.lens.copy()
        images = system.images.copy()

        # Prevent numerical issues at the origin
        lens[lens == 0]     = 1e-10
        images[images == 0] = 1e-10

        print(f'Plotting {name} …')
        graph_limits = mu.get_graph_limits(images)
        x_min, x_max = graph_limits[:, 0]

        # Iterate over the 4 possibilities of the “missing” image
        for missing_idx in range(4):
            img_missing  = images[missing_idx]
            imgs_given   = np.delete(images, missing_idx, axis=0)
            hyp_pts      = np.vstack((lens, imgs_given))
            hyp_coeffs   = mu.generate_hyperbola(hyp_pts)

            # Affine transform to XY space
            M, t, theta  = mu.hyperbola_xy_transform(hyp_coeffs)
            x0, y0       = t

            lens_xy      = (M @ lens.T).T + t
            img_missing_xy = M @ img_missing + t
            imgs_given_xy  = (M @ imgs_given.T).T + t

            k_param = lens_xy[0, 0] * lens_xy[1, 0]   # product xₗ yₗ in XY
            all_imgs_xy = np.vstack((imgs_given_xy, img_missing_xy))

            # Choose plotting frame
            if plot_transformed:
                graph_limits = mu.get_graph_limits(all_imgs_xy)

            # Solve ellipse(s) in XY space through the three known images
            ell_solutions = mu.generate_ellipse(imgs_given_xy, k_param)

            for sol_idx, (a, b, x_s) in enumerate(ell_solutions):
                y_s = k_param / x_s
                src_xy = np.array([x_s, y_s])
                src = np.linalg.inv(M) @ (src_xy - t)

                # Ellipse coefficients in original frame
                ell_coeffs = mu.XY_ellipse_to_xy_conic(
                    a, b, x_s, y_s, x0, y0, theta
                )

                xs, hyp_y1, hyp_y2 = mu.generate_conic_linspaces(
                    hyp_coeffs, 2000, x_min, x_max
                )
                _, ell_y1, ell_y2 = mu.generate_conic_linspaces(
                    ell_coeffs, 2000, x_min, x_max
                )

                if not hyp_y1.size or not ell_y1.size:
                    print('Graphing error: no curve points generated')
                    continue

                # Build graph ------------------------------------------------
                graph = gu.Graph(title=name, xaxis_title='RA', yaxis_title='Dec')
                graph.update_axes(xaxis_range=graph_limits[:, 0],
                                  yaxis_range=graph_limits[:, 1])

                hyp_x1, hyp_x2 = xs, xs
                ell_x1, ell_x2 = xs, xs
                lens_plot      = lens
                imgs_plot      = imgs_given
                miss_plot      = img_missing
                src_plot       = src

                if plot_transformed:
                    hyp_x1, hyp_y1 = mu.transform_curve(hyp_x1, hyp_y1, M, t)
                    hyp_x2, hyp_y2 = mu.transform_curve(hyp_x2, hyp_y2, M, t)
                    ell_x1, ell_y1 = mu.transform_curve(ell_x1, ell_y1, M, t)
                    ell_x2, ell_y2 = mu.transform_curve(ell_x2, ell_y2, M, t)

                    imgs_plot = imgs_given_xy
                    lens_plot = lens_xy.reshape(2)
                    miss_plot = img_missing_xy
                    src_plot  = src_xy

                # Plot curves and points ------------------------------------
                graph.add_trace(hyp_x1, hyp_y1, 'hyperbola')
                graph.add_trace(hyp_x2, hyp_y2, 'hyperbola', showlegend=False)
                graph.add_trace(ell_x1, ell_y1, 'ellipse')
                graph.add_trace(ell_x2, ell_y2, 'ellipse', showlegend=False)

                graph.add_trace(imgs_plot[:, 0], imgs_plot[:, 1], 'images')
                graph.add_trace(lens_plot[0], lens_plot[1], 'lens')
                graph.add_trace(src_plot[0],  src_plot[1],  'source')
                graph.add_trace(miss_plot[0], miss_plot[1], 'missing_image')
                graph.enable_legend()

                # Save figure -----------------------------------------------
                fname  = f'{name}[{missing_idx}]_{sol_idx}.png'
                subdir = 'transformed' if plot_transformed else 'original'
                out    = f'{SAVE_DIR}/{subdir}/{fname}'
                graph.save(out)


if __name__ == '__main__':
    print('Running in transformed frame' if PLOT_TRANSFORMED
          else 'Running in original frame')
    main(PLOT_TRANSFORMED)
