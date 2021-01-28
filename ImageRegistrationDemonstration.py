import os
import numpy as np
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import imageio
import imreg_dft as ird


from skimage import data
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift


image = data.camera()
shift = (-22.4, 53.32)
# The shift corresponds to the pixel offset relative to the reference image
offset_image = fourier_shift(np.fft.fftn(image), shift)
offset_image = np.fft.ifftn(offset_image)
print(f"Known offset (y, x): {shift}")


# pixel precision first
shift, _, _ = phase_cross_correlation(image, offset_image)
subshift, _, _ = phase_cross_correlation(image, offset_image, upsample_factor=100)
AlignedImage = fourier_shift(np.fft.fftn(offset_image), shift)
AlignedImage = np.fft.ifftn(AlignedImage)
subAlignedImage = fourier_shift(np.fft.fftn(offset_image), subshift)
subAlignedImage = np.fft.ifftn(subAlignedImage)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 5, 1)
ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 5, 3)
ax4 = plt.subplot(1, 5, 4, sharex=ax1, sharey=ax1)
ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

ax4.imshow(AlignedImage.real, cmap='gray')
ax4.set_axis_off()
ax4.set_title('Aligned image')

ax5.imshow(subAlignedImage.real, cmap='gray')
ax5.set_axis_off()
ax5.set_title('SubPixel Aligned image')

plt.show()

print(f"Detected pixel offset (y, x): {shift}")


