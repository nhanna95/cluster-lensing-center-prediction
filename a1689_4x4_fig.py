from pathlib import Path
from PIL import Image
import numpy as np
import fitsio
import matplotlib.pyplot as plt

import util.data as du

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_CENTROID = True  # use centroided images (True) or original FITS (False)

# I/O paths -----------------------------------------------------------------
if USE_CENTROID:
    INPUT_MASKS = [
        Path('fits/A1689/masks/centroid/individual/4_darkened_image_all.fits'),
        Path('fits/A1689/masks/centroid/individual/8_darkened_image_all.fits'),
        Path('fits/A1689/masks/centroid/individual/9_darkened_image_all.fits'),
        Path('fits/A1689/masks/centroid/individual/19_darkened_image_all.fits'),
    ]
else:
    INPUT_MASKS = [
        Path('fits/A1689/masks/coe/individual/4_darkened_image_all.fits'),
        Path('fits/A1689/masks/coe/individual/8_darkened_image_all.fits'),
        Path('fits/A1689/masks/coe/individual/9_darkened_image_all.fits'),
        Path('fits/A1689/masks/coe/individual/19_darkened_image_all.fits'),
    ]
    
IDS = ['4', '8', '9', '19']
OUTPUT_DIR = Path('insets/centroid') if USE_CENTROID else Path('insets/coe')

# Crop / intensity parameters ----------------------------------------------
CROP_HALF_WIDTH_PX = 30        # pixels (total crop side = 2 × half-width)
INTENSITY_CLIP = (0.84, 0.885)  # min, max before normalisation

HALF_WIDTHS = np.ones((4,2)) * CROP_HALF_WIDTH_PX  # half-widths for each quad
half_width = 30

# HALF_WIDTHS = np.array([
#     [15, 15],
#     [10, 30],
#     [20, 15],
#     [30, 30]
# ])

# Plot toggle ---------------------------------------------------------------
SHOW_PLOT = False

# Primary-image centroids (1-based FITS coords) -----------------------------

DATASET = du.a1689_FITS_quads_centroid if USE_CENTROID else du.a1689_FITS_quads
folder_name = 'centroid' if USE_CENTROID else 'coe'

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalise_to_uint8(arr, vmin, vmax):
    '''Clip *arr* to [vmin, vmax] and rescale to 0-255 uint8.'''
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-12)  # avoid /0
    return (arr * 255).astype(np.uint8)

def _crop(data, x, y, half):
    '''Return a square crop of side 2⋅half centred on (x, y).'''
    y0, y1 = max(y - half, 0), min(y + half, data.shape[0])
    x0, x1 = max(x - half, 0), min(x + half, data.shape[1])
    return data[y0:y1, x0:x1]

def output_filename(image_id, idx, half_width):
    '''Construct the JPEG output filename for a given quad and image index.'''
    return OUTPUT_DIR / 'individual' / f'{image_id}_{idx}_{half_width}.jpeg'

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    '''End-to-end processing routine.'''
    ensure_output_dir()

    for mask_path, quad, crop_widths, quad_id in zip(INPUT_MASKS, DATASET, HALF_WIDTHS, IDS):
        data, _ = fitsio.read(mask_path.as_posix(), header=True)
        print(f'Processing quad {quad_id} …')
        images = quad.images.copy()

        for img_idx, (x_c, y_c) in enumerate(images):
            # FITS → zero-based pixel origin
            x_px, y_px = int(x_c) - 1, int(y_c) - 1
            
            # half_width = int(crop_widths[img_idx])

            crop = _crop(data, x_px, y_px, half_width)
            crop_u8 = normalise_to_uint8(crop, *INTENSITY_CLIP)
            crop_u8 = np.flipud(crop_u8)  # match original orientation

            outfile = output_filename(quad_id, img_idx, half_width)
            Image.fromarray(crop_u8).save(outfile)
            print(f'  ⤷ Saved {outfile}')

            if SHOW_PLOT:
                plt.figure(figsize=(2, 2))
                plt.imshow(crop, cmap='gray', origin='lower')
                plt.title(f'Quad {quad_id} - image {img_idx}')
                plt.axis('off')
                plt.tight_layout()

    if SHOW_PLOT: plt.show()
        
    print('Done.')

if __name__ == '__main__':
    main()