from pathlib import Path
from PIL import Image
import numpy as np
import fitsio
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

data_set = 'coe'  # 'lim', 'centroid', or 'coe'

# I/O paths -----------------------------------------------------------------
if data_set == 'centroid':
    INPUT_MASKS = [
        Path('fits/A1689/masks/centroid/individual/4_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/centroid/individual/8_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/centroid/individual/9_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/centroid/individual/19_darkened_image_ellipse_major.fits'),
    ]
elif data_set == 'coe':
    INPUT_MASKS = [
        Path('fits/A1689/masks/coe/individual/4_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/coe/individual/8_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/coe/individual/9_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/coe/individual/19_darkened_image_ellipse_major.fits'),
    ]
else:
    INPUT_MASKS = [
        Path('fits/A1689/masks/lim/individual/4_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/lim/individual/8_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/lim/individual/9_darkened_image_ellipse_major.fits'),
        Path('fits/A1689/masks/lim/individual/19_darkened_image_ellipse_major.fits'),
    ]
    
IDS = ['4', '8', '9', '19']
OUTPUT_DIR = Path(f'insets/{data_set}')  # output directory for cropped images

# Crop / intensity parameters ----------------------------------------------
CROP_HALF_WIDTH_PX = 30        # pixels (total crop side = 2 × half-width)
INTENSITY_CLIP = (0.84, 0.885)  # min, max before normalisation

HALF_WIDTHS = np.ones((4,2)) * CROP_HALF_WIDTH_PX  # half-widths for each quad

HALF_WIDTHS = np.array([
    [15, 15],
    [10, 30],
    [20, 15],
    [30, 30]
])

# Plot toggle ---------------------------------------------------------------
SHOW_PLOT = False

# Primary-image centroids (1-based FITS coords) -----------------------------
CENTROID_PRIMARY = np.array([
    [[2465.656, 3016.3897], [3807.5621, 2473.7682]],
    [[2243.8931, 2898.4897], [4030.3778, 2776.3938]],
    [[2601.6592, 3408.1493], [1636.1257, 2172.3832]],
    [[2202.6910, 2727.1021], [4118.7687, 2781.5778]]
], dtype=float)

COE_PRIMARY = np.array([
    [[2463.7202,3015.7972], [3805.2522,2472.7942]],
    [[2242.6801,2898.7967], [4032.8922,2780.5928]],
    [[2600.1827,3407.7973], [1635.6501,2172.3939]]
], dtype=float)

LIM_PRIMARY = np.array([
    [[2463.6,3019.0372], [3804.5921,2478.3143]],
    [[2241.1808,2901.6767], [4031.3322,2777.8328]],
    [[2601.8022,3414.3173], [1640.8694,2173.0339]],
    [[2201.5921,2729.5965], [4114.8296,2785.7522]]
], dtype=float)

if data_set == 'centroid':
    DATASET = CENTROID_PRIMARY
elif data_set == 'coe':
    DATASET = COE_PRIMARY
else:
    DATASET = LIM_PRIMARY

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

    for mask_path, quad_centroids, crop_widths, quad_id in zip(INPUT_MASKS, DATASET, HALF_WIDTHS, IDS):
        data, _ = fitsio.read(mask_path.as_posix(), header=True)
        print(f'Processing quad {quad_id} …')

        for img_idx, (x_c, y_c) in enumerate(quad_centroids):
            # FITS → zero-based pixel origin
            x_px, y_px = int(x_c) - 1, int(y_c) - 1
            
            half_width = int(crop_widths[img_idx])

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

def alt():
    '''Alternative processing routine for net image major.'''
    mask_folder = Path(f'fits/A1689/masks/{data_set}')
    mask_path = mask_folder / 'net_image_major.fits'
    data, _ = fitsio.read(mask_path.as_posix(), header=True)
    data = np.flipud(data)

    INTENSITY_CLIP_ALT = (0.6, 0.915)
    data = np.clip(data, *INTENSITY_CLIP_ALT)

    center = np.array([2650, 2550])
    half_width = 1730
    y0, y1 = center[1] - half_width, center[1] + half_width
    x0, x1 = center[0] - half_width, center[0] + half_width
    cropped_data = data[y0:y1, x0:x1]

    cropped_data = normalise_to_uint8(cropped_data, *INTENSITY_CLIP_ALT)
    cropped_data = np.nan_to_num(cropped_data, nan=0)

    outfile = OUTPUT_DIR / 'net_image_major_cropped.jpeg'
    Image.fromarray(cropped_data).save(outfile)
    print(f'Saved {outfile}')

    if SHOW_PLOT:
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_data, cmap='gray', origin='lower')
        plt.title('Net Image Major - Cropped')
        plt.axis('off')
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    # main()
    alt()