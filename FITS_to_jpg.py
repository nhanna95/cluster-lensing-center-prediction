from pathlib import Path
from PIL import Image
import numpy as np
import fitsio
import matplotlib.pyplot as plt

import util.data as du

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLUSTER = du.clusters.get('clj2325') # 'a1689' or 'clj2325'
SYSTEMS = CLUSTER.systems

FILE_NAME = 'net_image_major_circles' # 'net_image_major', 'net_image_major_circles'

# Flip options
FLIP_HOR = False
FLIP_VER = True

# Intensity scaling options
SCALE = 'squared' # 'sinh', 'log', 'sqrt', 'power', 'linear', 'asinh', 'squared'
INTENSITY_CLIP = (0.0, 0.973706)  # min, max before normalisation

# Crop parameters
CENTER = np.array([3411, 3869])
HALF_WIDTH = 1400

MASK_FOLDER = Path(f'fits/{CLUSTER.folder_path}/masks')
MASK_PATH = MASK_FOLDER / f'{FILE_NAME}.fits'
OUTPUT_DIR = Path(f'insets/{CLUSTER.folder_path}')  # output directory for cropped images

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalise_to_uint8(arr, vmin, vmax):
    '''Clip *arr* to [vmin, vmax] and rescale to 0-255 uint8.'''
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-12)  # avoid /0
    return (arr * 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    '''End-to-end processing routine.'''
    data, _ = fitsio.read(MASK_PATH.as_posix(), header=True)
    data = np.nan_to_num(data, nan=1)
    
    if FLIP_HOR:
        data = np.fliplr(data)
        CENTER[0] = data.shape[1] - CENTER[0]
    if FLIP_VER:
        data = np.flipud(data)
        CENTER[1] = data.shape[0] - CENTER[1]
    

    y0, y1 = CENTER[1] - HALF_WIDTH, CENTER[1] + HALF_WIDTH
    x0, x1 = CENTER[0] - HALF_WIDTH, CENTER[0] + HALF_WIDTH
    cropped_data = data[y0:y1, x0:x1]

    # Apply intensity scaling
    if SCALE == 'log':
        cropped_data = np.log10(cropped_data + 1e-12)
    elif SCALE == 'sqrt':
        cropped_data = np.sqrt(cropped_data)
    elif SCALE == 'asinh':
        cropped_data = np.arcsinh(cropped_data)
    elif SCALE == 'power':
        cropped_data = np.power(cropped_data, 0.5)
    elif SCALE == 'squared':
        cropped_data = np.square(cropped_data)
    elif SCALE == 'sinh':
        cropped_data = np.sinh(cropped_data)
    elif SCALE == 'linear':
        pass # no change
    else:
        raise ValueError(f'Unknown SCALE option: {SCALE}')

    cropped_data = np.clip(cropped_data, *INTENSITY_CLIP)
    cropped_data = normalise_to_uint8(cropped_data, *INTENSITY_CLIP)

    outfile = OUTPUT_DIR / f'{FILE_NAME}_cropped.jpeg'
    Image.fromarray(cropped_data).save(outfile)
    print(f'Saved {outfile}')

if __name__ == '__main__':
    main()