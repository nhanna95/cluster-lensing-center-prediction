import numpy as np
import util.data as du

def ra_dec_to_xy(coords, center):
    """
    Convert RA/Dec coordinates to X/Y coordinates based on a center point.
    
    :param coords: List of [RA, Dec] coordinates.
    :param center: Center point as [RA, Dec].
    :return: List of [X, Y] coordinates.
    """
    c_ra, c_dec = np.deg2rad(center)
    coords = np.array(coords)
    ra, dec = np.deg2rad(coords[:, 0]), np.deg2rad(coords[:, 1])

    X = np.cos(dec) * np.sin(ra - c_ra) / (np.cos(c_dec) * np.cos(dec) * np.cos(ra - c_ra) + np.sin(c_dec) * np.sin(dec))
    Y = (np.sin(c_dec) * np.cos(dec) * np.cos(ra - c_ra) - np.cos(c_dec) * np.sin(dec)) / (np.cos(c_dec) * np.cos(dec) * np.cos(ra - c_ra) + np.sin(c_dec) * np.sin(dec))

    return np.column_stack((X, Y))

if __name__ == '__main__':
    center = [151.14144831, 41.21367857]
    data = du.SDSS_J1004_ra_dec
    coords = [coord for system in data for coord in system.images]
    xy_coords = ra_dec_to_xy(coords, center)
    print(xy_coords)