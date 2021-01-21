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

# basedir = os.path.join('/home/mbraun/XRDProcessing/imreg_dft-master/resources/', 'examples')
# # the TEMPLATE
# im0 = imageio.imread(os.path.join(basedir, "sample1.png"))
# # the image to be transformed
# im1 = imageio.imread(os.path.join(basedir, "sample3.png"))
# result = ird.similarity(im0, im1, numiter=3)
#
# assert "timg" in result
# # Maybe we don't want to show plots all the time
# ird.imshow(im0, im1, result['timg'])
# plt.show()


image = data.camera()
shift = (-22.4, 53.32)
# The shift corresponds to the pixel offset relative to the reference image
offset_image = fourier_shift(np.fft.fftn(image), shift)
offset_image = np.fft.ifftn(offset_image)
print(f"Known offset (y, x): {shift}")


# pixel precision first
shift, error, diffphase = phase_cross_correlation(image, offset_image)
AlignedImage = fourier_shift(np.fft.fftn(offset_image), shift)
AlignedImage = np.fft.ifftn(AlignedImage)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 4, 1)
ax2 = plt.subplot(1, 4, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 4, 3)
ax4 = plt.subplot(1, 4, 4, sharex=ax1, sharey=ax1)

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

plt.show()

print(f"Detected pixel offset (y, x): {shift}")

# subpixel precision
shift, error, diffphase = phase_cross_correlation(image, offset_image,
                                                  upsample_factor=100)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

plt.show()

