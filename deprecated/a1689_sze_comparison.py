import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

xray_center = np.array([197.8725, -1.3414])
sze_center = np.array([197.8727083, -1.3411389])

estimated_center = np.array([197.87560, -1.34195])

five_arcmin = 5 / 60  # degrees

extent = [sze_center[0] - five_arcmin, sze_center[0] + five_arcmin,
          sze_center[1] - five_arcmin, sze_center[1] + five_arcmin]


visual_xray_center = np.array([197.87267825, -1.34041916168])
plotted_xray_center = np.array([197.868534052])

print(extent)

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

def main():
    image = mpimg.imread("sze_data.png").copy()

    crop_x_start_pixel = int(crop_x_start * image.shape[1])
    crop_x_end_pixel = int(crop_x_end * image.shape[1])
    crop_y_start_pixel = int(crop_y_start * image.shape[0])
    crop_y_end_pixel = int(crop_y_end * image.shape[0])

    cropped_image = image#[crop_y_start_pixel:crop_y_end_pixel, crop_x_start_pixel:crop_x_end_pixel]
    cropped_extent = [
        extent[0] + crop_x_start * (extent[1] - extent[0]),
        extent[0] + crop_x_end * (extent[1] - extent[0]),
        extent[3] - crop_y_end * (extent[3] - extent[2]),
        extent[3] - crop_y_start * (extent[3] - extent[2])
    ]
    
    print(cropped_extent)
    
    cropped_image = np.fliplr(cropped_image)

    plt.imshow(cropped_image, extent=cropped_extent, aspect='equal')
    ax = plt.gca()
    # Invert the x-axis
    ax.invert_yaxis()
    plt.scatter(xray_center[0], xray_center[1], marker='s', color='black', s=30, label='X-ray Center')
    plt.scatter(sze_center[0], sze_center[1], marker='D', color='black', s=25, label='SZE Center')
    plt.scatter(estimated_center[0], estimated_center[1], marker='o', color='black', s=25, label='Estimated Center')
    
    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title("A1689: SZ Effect")
    
    plt.savefig("plots/comparison/sze_comparison_new.png", dpi=600)
    plt.show()
    
    
if __name__ == "__main__":
    main()
