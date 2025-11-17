import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

xray_center = np.array([197.8725, -1.3414])
sze_center = np.array([197.8727083, -1.3411389])

estimated_center = np.array([197.87560, -1.34195])

bump = np.array([0.0028, 0.0032])

# xray_center = xray_center + bump
# sze_center = sze_center + bump
# estimated_center = estimated_center + bump

extent_nonbump = [197.6666, 198, -1.1666, -1.5]  # RA_min, RA_max, Dec_min, Dec_max

extent = [extent_nonbump[0] - bump[0], extent_nonbump[1] - bump[0],
          extent_nonbump[2] - bump[1], extent_nonbump[3] - bump[1]]

min_RA, max_RA = extent[0], extent[1]
min_Dec, max_dec = extent[2], extent[3]

min_RA = 197.8
max_RA = 197.95
min_Dec = -1.3
max_dec = -1.4

crop_x_start = (min_RA - extent[0]) / (extent[1] - extent[0])
crop_x_end = (max_RA - extent[0]) / (extent[1] - extent[0])
crop_y_start = (extent[3] - max_dec) / (extent[3] - extent[2])
crop_y_end = (extent[3] - min_Dec) / (extent[3] - extent[2])

# crop_x_start = 0.35
# crop_x_end = 0.9
# crop_y_start = 0.25
# crop_y_end = 0.7

def main():
    image = mpimg.imread("comparison_data/xray_data.png").copy()

    crop_x_start_pixel = int(crop_x_start * image.shape[1])
    crop_x_end_pixel = int(crop_x_end * image.shape[1])
    crop_y_start_pixel = int(crop_y_start * image.shape[0])
    crop_y_end_pixel = int(crop_y_end * image.shape[0])

    cropped_image = image[crop_y_start_pixel:crop_y_end_pixel, crop_x_start_pixel:crop_x_end_pixel]
    cropped_extent = [
        extent[0] + crop_x_start * (extent[1] - extent[0]),
        extent[0] + crop_x_end * (extent[1] - extent[0]),
        extent[3] - crop_y_end * (extent[3] - extent[2]),
        extent[3] - crop_y_start * (extent[3] - extent[2])
    ]
    print(cropped_extent)

    plt.imshow(cropped_image, extent=cropped_extent, aspect='equal')
    plt.scatter(xray_center[0], xray_center[1], marker='s', color='black', s=30, label='X-ray Center')
    plt.scatter(sze_center[0], sze_center[1], marker='D', color='black', s=25, label='SZE Center')
    plt.scatter(estimated_center[0], estimated_center[1], marker='o', color='black', s=25, label='Estimated Center')
    
    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title("A1689: XMM X-ray Emission")
    
    plt.savefig("plots/comparison/xray_comparison_new.png", dpi=300)
    plt.show()
    
    
if __name__ == "__main__":
    main()
