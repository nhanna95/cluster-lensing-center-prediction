import numpy as np
import fitsio
import matplotlib.pyplot as plt


crop = np.array([41046, 7392, 41065, 7405]) # 41047, 7393, 41065, 7405

raw_data, hdr = fitsio.read('Compton-y Map Download.fits', header=True)

dtype      = raw_data.dtype
h, w       = raw_data.shape
min_val    = 0
max_val    = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0

xray_center = np.array([197.8725, -1.3414])
sze_center = np.array([197.8727083, -1.3411389])

estimated_center = np.array([197.87560, -1.34195])

print(h,w)
data = raw_data[crop[1]:crop[3], crop[0]:crop[2]]
shape = data.shape
print(shape)

upscaling_factor = 20

upscaled_data = np.zeros((shape[0]*upscaling_factor, shape[1]*upscaling_factor))
upscaled_shape = upscaled_data.shape

for i in range(shape[0]):
    for j in range(shape[1]):
        upscaled_data[i*upscaling_factor:(i+1)*upscaling_factor, j*upscaling_factor:(j+1)*upscaling_factor] = data[i, j]

gaussian_kernel = np.outer(np.exp(-np.linspace(-3, 3, upscaling_factor*2)**2 / 2), 
                          np.exp(-np.linspace(-3, 3, upscaling_factor*2)**2 / 2))
gaussian_kernel = 100 * gaussian_kernel / gaussian_kernel.sum()

# Output matrix dimensions
output_h = upscaled_data.shape[0] - gaussian_kernel.shape[0] + 1
output_w = upscaled_data.shape[1] - gaussian_kernel.shape[1] + 1

# Initialize output matrix
output_matrix = np.zeros((output_h, output_w))

# Perform convolution
for i in range(output_h):
    for j in range(output_w):
        region = upscaled_data[i:i+gaussian_kernel.shape[0], j:j+gaussian_kernel.shape[1]]
        output_matrix[i, j] = np.sum(region * gaussian_kernel)
        
        
shape = output_matrix.shape
print(shape)
final_crop = [int(upscaling_factor*0.2), int(upscaling_factor*0.5), int(shape[1]-upscaling_factor*0.5), int(shape[0]-upscaling_factor*0.5)]
print(final_crop)
# output_matrix = output_matrix[final_crop[1]:final_crop[3], final_crop[0]:final_crop[2]]

# output_matrix = np.flipud(output_matrix)
output_matrix = np.fliplr(output_matrix)

plt.imshow(output_matrix, cmap='plasma', extent=[197.8, 197.95, -1.3, -1.4])

plt.scatter(xray_center[0], xray_center[1], marker='s', color='black', s=30, label='X-ray Center')
plt.scatter(sze_center[0], sze_center[1], marker='D', color='black', s=25, label='SZE Center')
plt.scatter(estimated_center[0], estimated_center[1], marker='o', color='black', s=25, label='Estimated Center')
    

plt.xlabel("R.A. (deg)")
plt.ylabel("Dec. (deg)")
plt.title("A1689: SZ Effect")

plt.savefig("sze_comparison.png", dpi=600)
plt.show()

# plt.figure(figsize=(10, 8))
# plt.imshow(output_matrix, cmap='viridis', origin='lower')
# plt.colorbar(label='Pixel Value')
# plt.title('Cropped Image')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.savefig('output_matrix.png', dpi=300, bbox_inches='tight')
# plt.close()