from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import numpy as np

from tqdm import tqdm

import util.data as du
import util.graphing as gu
import util.math as mu

hdr  = fits.getheader('fits/A1689/original_data/hst_11710_05_acs_wfc_f814w_jb2g05_drc.fits', ext=1)
wcs = WCS(hdr)

standard_data = pd.read_pickle('a1689_standard_data.pkl')
bounding_box = pd.read_pickle('a1689_bounding_box.pkl')

standard_hyper_coeffs = np.vstack(standard_data['hyper_coeffs'].values)
standard_ellipse_coeffs = np.vstack(standard_data['ell_coeffs'].values)
standard_asymptotes = np.vstack(standard_data['asymptotes'].values)
standard_asymptotes = standard_asymptotes.reshape((4, -1, 2))
standard_side_flags = np.vstack(standard_data['side_flags'].values)
CENTER_OFFSET = standard_data['center_offset'].values[0]
x_min = bounding_box['x_min'].values[0]
x_max = bounding_box['x_max'].values[0]
y_min = bounding_box['y_min'].values[0]
y_max = bounding_box['y_max'].values[0]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = 'fits_lim' # 'fits' or 'fits_lim'

PARALLEL_NOISE_ONLY = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_box(points, x_min, x_max, y_min, y_max):
    '''Boolean mask selecting points inside the current bounding box.'''
    x, y = points[:, 0], points[:, 1]
    return (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    systems = du.quad_datasets.get(DATASET, du.a1689_FITS_quads_lim)
    parallel_noise = np.linspace(-500, 500, 51)
    
    results_df = pd.DataFrame({
        'quad_idx': [], 'image_id': [], 'noise': [],
        'x': [], 'y': [], 'ra': [], 'dec': [],
        'asym_angle': [], 'axis_ratio': [],
        'center_x': [], 'center_y': [],
        'error': [],
        'ell_center_x': [], 'ell_center_y': [],
        'nearest_x': [], 'nearest_y': [],
        'ell_center_ra': [], 'ell_center_dec': [],
        'nearest_ra': [], 'nearest_dec': []
    })
    
    # ---------------------------------------------------------------------
    # Per-system fits and layer plotting
    # ---------------------------------------------------------------------
    for quad_idx, system in enumerate(systems):
        print(f'Processing {system.name}…')
        name = system.name
        images = system.images.copy() - CENTER_OFFSET
        for i in range(4):
            print(f'  ⤷ Image {i + 1}…')
            tangent_slope = mu.ellipse_tangent_slope(standard_ellipse_coeffs[i], images[i])
            angle = np.arctan(tangent_slope)
            for noise in tqdm(parallel_noise):
                temp_images = images.copy()
                hyperbolas = standard_hyper_coeffs
                asymptotes = standard_asymptotes
                side_flags = standard_side_flags
                
                noise_x = noise * np.cos(angle)
                noise_y = noise * np.sin(angle)
                temp_images[i] += np.array([noise_x, noise_y])
                
                temp_images[temp_images == 0] = 1e-10                   # avoid exact zeros
            
                temp_images_wcs = wcs.all_pix2world(
                    temp_images[:, 0] + CENTER_OFFSET + 1, 
                    temp_images[:, 1] + CENTER_OFFSET + 1, 0)
                
                # Hyperbola fit
                hyp = mu.generate_hyperbola(temp_images)
                hyperbolas[quad_idx] = np.array(hyp)

                (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
                asymptotes[quad_idx] = [[m1, b1], [m2, b2]]
                
                angle1 = np.arctan(m1)
                angle2 = np.arctan(m2)
                
                asym_angle = angle1 if abs(angle1) > abs(angle2) else angle2
                
                # Keep only major branches
                try:
                    flags = mu.get_major_side_flag(temp_images, m1, b1, m2, b2)
                except:
                    print(f'  ⤷ Warning: No major branches found for quad {quad_idx}, image {i}. Skipping.')
                    continue
                side_flags[quad_idx] = flags

                # Ellipse through quad (for optional plotting / error estimate)
                ell_coeffs = mu.generate_optimal_ellipse(temp_images, hyp)
                ell_center = mu.find_conic_center(ell_coeffs)
                try:
                    axis_ratio = mu.ellipse_axis_ratio(ell_coeffs)
                except:
                    print(f'  ⤷ Warning: Could not compute axis ratio for quad {quad_idx}, image {i}. Skipping.')
                    continue
                error_distance, nearest_point = mu.point_to_hyperbola_distance(ell_center[0], ell_center[1], hyp, return_nearest_point=True)

                ell_center_wcs = wcs.all_pix2world(
                    ell_center[0] + CENTER_OFFSET + 1, 
                    ell_center[1] + CENTER_OFFSET + 1, 0)
                
                nearest_point_wcs = wcs.all_pix2world(
                    nearest_point[0] + CENTER_OFFSET + 1, 
                    nearest_point[1] + CENTER_OFFSET + 1, 0)

                # ---------------------------------------------------------------------
                # Pairwise hyperbola intersections
                # ---------------------------------------------------------------------
                inter_pts = []
                
                for j in range(len(hyperbolas)):
                    for k in range(j + 1, len(hyperbolas)):
                        pts = mu.conic_intersections(hyperbolas[j], hyperbolas[k])
                        if pts.size == 0:
                            continue

                        ms = (
                            asymptotes[j][0][0], asymptotes[j][1][0],
                            asymptotes[k][0][0], asymptotes[k][1][0],
                        )
                        bs = (
                            asymptotes[j][0][1], asymptotes[j][1][1],
                            asymptotes[k][0][1], asymptotes[k][1][1],
                        )
                        flags = (
                            side_flags[j][0], side_flags[j][1],
                            side_flags[k][0], side_flags[k][1],
                        )

                        pts = pts[mu.create_side_flag_mask(
                            pts[:, 0], pts[:, 1], ms, bs, flags
                        )]
                        if pts.size == 0:
                            continue

                        inter_pts.append(pts)

                inter_pts = np.vstack(inter_pts)

                # Bounding-box filter
                mask = _in_box(inter_pts, x_min, x_max, y_min, y_max)
                inter_pts = inter_pts[mask]

                predicted = np.average(inter_pts, axis=0)
                
                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({
                        'quad_idx': [quad_idx],
                        'image_id': [i],
                        'noise': [noise],
                        'x': [temp_images[i][0] + CENTER_OFFSET],
                        'y': [temp_images[i][1] + CENTER_OFFSET],
                        'ra': [temp_images_wcs[0][i]],
                        'dec': [temp_images_wcs[0][i]],
                        'asym_angle': [asym_angle],
                        'axis_ratio': [axis_ratio],
                        'center_x': [predicted[0] + CENTER_OFFSET],
                        'center_y': [predicted[1] + CENTER_OFFSET],
                        'error': [error_distance],
                        'ell_center_x': [ell_center[0]],
                        'ell_center_y': [ell_center[1]],
                        'nearest_x': [nearest_point[0] + CENTER_OFFSET],
                        'nearest_y': [nearest_point[0] + CENTER_OFFSET],
                        'ell_center_ra': [ell_center_wcs[0]],
                        'ell_center_dec': [ell_center_wcs[1]],
                        'nearest_ra': [nearest_point_wcs[0]],
                        'nearest_dec': [nearest_point_wcs[1]]
                    })
                ], ignore_index=True)
            
    results_df.to_pickle(f'results_500_{DATASET}.pkl')
    
    
def alt():
    systems = du.quad_datasets.get(DATASET, du.a1689_FITS_quads_lim)
    parallel_noise = np.linspace(-500, 500, 101)
    
    results_df = pd.DataFrame({
        'quad_idx': [], 'image_id': [], 
        'x': [], 'y': [], 'displacement': [],
        'asym_angle': [], 'axis_ratio': [], 'ell_coeffs': []
    })
    
    # ---------------------------------------------------------------------
    # Per-system fits and layer plotting
    # ---------------------------------------------------------------------
    for quad_idx, system in enumerate(systems):
        print(f'Processing {system.name}…')
        name = system.name
        images = system.images.copy() - CENTER_OFFSET
        for i in range(4):
            print(f'  ⤷ Image {i + 1}…')
            tangent_slope = mu.ellipse_tangent_slope(standard_ellipse_coeffs[i], images[i])
            angle = np.arctan(tangent_slope)
            for noise in tqdm(parallel_noise):
                temp_images = images.copy()
                
                noise_x = noise * np.cos(angle)
                noise_y = noise * np.sin(angle)
                temp_images[i] += np.array([noise_x, noise_y])
                
                temp_images[temp_images == 0] = 1e-10                   # avoid exact zeros
            
                # Hyperbola fit
                hyp = mu.generate_hyperbola(temp_images)
                
                (m1, b1), (m2, b2) = mu.get_asymptotes(hyp)
                
                angle1 = np.arctan(m1)
                angle2 = np.arctan(m2)
                
                asym_angle = angle1 if abs(angle1) > abs(angle2) else angle2
                
                # Ellipse through quad (for optional plotting / error estimate)
                ell_coeffs = mu.generate_optimal_ellipse(temp_images, hyp)
                try:
                    axis_ratio = mu.ellipse_axis_ratio(ell_coeffs)
                except:
                    print(f'  ⤷ Warning: Could not compute axis ratio for quad {quad_idx}, image {i}. Skipping.')
                    continue

                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({
                        'quad_idx': [quad_idx],
                        'image_id': [i],
                        'x': [temp_images[i][0]],
                        'y': [temp_images[i][1]],
                        'displacement': [noise],
                        'asym_angle': [asym_angle],
                        'axis_ratio': [axis_ratio],
                        'ell_coeffs': [ell_coeffs]
                    })
                ], ignore_index=True)
            
    results_df.to_pickle(f'ellipse_results_500_{DATASET}.pkl')


if __name__ == '__main__':
    # main()
    alt()
