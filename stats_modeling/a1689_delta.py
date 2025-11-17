import numpy as np

import util.data as du
import util.math as mu
import pandas as pd

def main():
    systems = du.a1689_all_fits

    fit_errors = []
    major_axes = []

    for idx, system in enumerate(systems):
        name = system.name
        images = system.images.copy()

        images[images == 0] = 1e-10

        # Hyperbola fit
        hyp = mu.generate_hyperbola(images)
        ell_coeffs = mu.generate_optimal_ellipse(images, hyp)
        ell_center = mu.find_conic_center(ell_coeffs)
        
        major_axis = mu.ellipse_major_axis(ell_coeffs)
        
        major_axes.append(major_axis)
        
        fit_errors.append(
            mu.point_to_hyperbola_distance(ell_center[0], ell_center[1], hyp)
        )
        # print(f'Quad {name} fit error: {fit_errors[-1]:.2f} pixels, major axis: {major_axis:.2f} pixels')
        
    names = [sys.name[:-4] for sys in systems]
    fit_errors = np.array(fit_errors)
    major_axes = np.array(major_axes)
    
    df = pd.DataFrame({
        'name': names,
        'fit_error': fit_errors,
        'major_axis': major_axes,
        'normalized': fit_errors / major_axes
    })
    df = df.sort_values(by='normalized', ascending=True)
    print(df)
    print('-' * 80)
    df = df.sort_values(by='fit_error', ascending=True)
    print(df)

if __name__ == '__main__':
    main()
