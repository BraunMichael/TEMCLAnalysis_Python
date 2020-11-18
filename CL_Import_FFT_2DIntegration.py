import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, Slider, PolygonSelector
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString, MultiLineString, Polygon

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import gaussian_filter

from Utility.peak_prominence2d import *

# User adjustable parameters
pixelScale = 49  # nm per pixel
averagedSlices = 5  # Averaged wavelengths per center wavelength (symmetric)
rawCL = np.loadtxt('5min_Sample2.txt')
# rawCL = np.loadtxt('CL Spectrum Image_12mW_3min_5_feb10.txt')


def radial_profile(data, center):
    # https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def detect_peaks(image, neighborhoodSize):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    # local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max = maximum_filter(image, size=neighborhoodSize) == image

    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image == 0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


class ImageHandler:
    def __init__(self):
        self.imageMask = None
        self.boundaryPoly = None
        self.xMin = 0
        self.yMin = 0
        self.xMax = 1
        self.yMax = 1


class FFTManager:
    def __init__(self, fig, ax):
        self.coords = {}
        self.selectedPolygon = None
        self.fig = fig
        self.ax = ax

    def FFTButtonClicked(self, _):
        print(self.coords)
        if len(self.ax.patches) > 2:
            self.ax.patches[-1].remove()
        rect = RectangleSelector(self.ax, self.RangeSelection, drawtype='box', rectprops=dict(facecolor='red', edgecolor='none', alpha=0.3, fill=True))
        plt.close()

    def RangeSelection(self, eclick, erelease):
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        self.coords['x'] = [eclick.xdata, erelease.xdata]
        self.coords['y'] = [eclick.ydata, erelease.ydata]
        self.selectedPolygon = Polygon([(eclick.xdata, eclick.ydata), (erelease.xdata, eclick.ydata), (erelease.xdata, erelease.ydata), (eclick.xdata, erelease.ydata)])
        if len(self.ax.patches) > 1:
            self.ax.patches[-1].remove()
        selection = mpatches.Rectangle((eclick.xdata, eclick.ydata), abs(eclick.xdata - erelease.xdata), abs(eclick.ydata - erelease.ydata), linewidth=1, edgecolor='green', facecolor='none', alpha=0.5, fill=False)
        self.ax.add_patch(selection)
        self.fig.canvas.draw()





wavelengthsRaw = np.loadtxt('Spectrum_WavelengthInfo.txt', delimiter=', ')
wavelengths = wavelengthsRaw[:, 0]
assert len(rawCL) % len(wavelengths) == 0, "Your CL data is not an even multiple of your number of wavelengths, you probably need an updated wavelengths file."
frameNum = int(len(wavelengths)/2)

assert averagedSlices % 2 == 1, "Only odd numbers of averaged wavelengths allowed to simplify calculations/meaning of averaged wavelengths"
out = np.reshape(rawCL.flatten(), (len(wavelengths), int(rawCL.shape[0]/len(wavelengths)), rawCL.shape[1]))
outAveraged = np.zeros((len(wavelengths), int(rawCL.shape[0]/len(wavelengths)), rawCL.shape[1]))
for centerSlice in range(len(wavelengths)):
    lowerSlice = max(0, centerSlice - int(((averagedSlices - 1) / 2)))
    upperSlice = min(len(wavelengths)-1, centerSlice + int(((averagedSlices - 1) / 2)))
    outAveraged[centerSlice, :, :] = np.mean(out[lowerSlice:upperSlice+1, :, :], 0)
outAveraged = outAveraged + abs(np.min(outAveraged)) + 0.001


fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
CLimage = plt.imshow(outAveraged[frameNum, :, :], interpolation='none', vmin=np.min(outAveraged[frameNum, :, :]), vmax=np.max(outAveraged[frameNum, :, :]),
                     cmap='plasma', norm=LogNorm())
plt.subplots_adjust(bottom=0.18)
ax.margins(x=0)
plt.axis('equal')
axSlice = plt.axes([0.25, 0.1, 0.65, 0.03])
sSlice = Slider(axSlice, 'Wavelength (nm)', 0, len(wavelengths)-1, valinit=int(len(wavelengths)/2), valfmt='%0.0f')
sSlice.valtext.set_text(int(wavelengths[int(len(wavelengths)/2)]))


def update(_):
    sSlice.valtext.set_text(int(wavelengths[int(sSlice.val)]))
    CLimage.set_data(outAveraged[int(sSlice.val), :, :])
    CLimage.vmin = np.min(outAveraged[int(sSlice.val), :, :])
    CLimage.vmax = np.max(outAveraged[int(sSlice.val), :, :])
    fig.canvas.draw_idle()


sSlice.on_changed(update)

imageHandler = ImageHandler()
imageHandler.xMax = outAveraged.shape[1]
imageHandler.yMax = outAveraged.shape[2]


def PolySelection(polygonVertices):
    polygonVertices = [(min(max(0, entry[0]), imageHandler.xMax), min(max(0, entry[1]), imageHandler.yMax)) for entry in polygonVertices]
    xPoints, yPoints = np.meshgrid(np.arange(outAveraged.shape[2]), np.arange(outAveraged.shape[1]))
    xPoints, yPoints = xPoints.flatten(), yPoints.flatten()

    points = np.vstack((xPoints, yPoints)).T
    polyPath = Path(polygonVertices)
    imageMask = polyPath.contains_points(points)
    imageHandler.imageMask = imageMask.reshape((outAveraged.shape[1], outAveraged.shape[2]))
    imageHandler.boundaryPoly = polyPath


poly = PolygonSelector(ax, PolySelection)

plt.show()
plt.close()
#
#
outAveraged = outAveraged * imageHandler.imageMask + 1

fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
CLimage = plt.imshow(outAveraged[frameNum, :, :], interpolation='none', vmin=np.min(outAveraged[frameNum, :, :]), vmax=np.max(outAveraged[frameNum, :, :]), cmap='plasma', norm=LogNorm())
plt.subplots_adjust(bottom=0.18)
ax.margins(x=0)
plt.axis('equal')
axSlice = plt.axes([0.25, 0.1, 0.65, 0.03])
sSlice = Slider(axSlice, 'Wavelength (nm)', 0, len(wavelengths)-1, valinit=int(len(wavelengths)/2), valfmt='%0.0f')
sSlice.valtext.set_text(int(wavelengths[int(len(wavelengths)/2)]))


def update(_):
    sSlice.valtext.set_text(int(wavelengths[int(sSlice.val)]))
    CLimage.set_data(outAveraged[int(sSlice.val), :, :])
    CLimage.vmin = np.min(outAveraged[int(sSlice.val), :, :])
    CLimage.vmax = np.max(outAveraged[int(sSlice.val), :, :])
    fig.canvas.draw_idle()


sSlice.on_changed(update)

fftManager = FFTManager(fig, ax)
axFFT = plt.axes([0.7, 0.02, 0.2, 0.075])
bFFT = Button(axFFT, 'Calc FFT')
rect = RectangleSelector(ax, fftManager.RangeSelection, drawtype='box', rectprops=dict(facecolor='none', edgecolor='red', alpha=0.5, fill=False))
bFFT.on_clicked(fftManager.FFTButtonClicked)

plt.show()
centerSliceIndex = int(sSlice.val)
centerWavelengthValue = wavelengths[centerSliceIndex]
coordsOut = fftManager.coords
plt.close()
xMin = int(round(coordsOut['x'][0]))
xMax = int(round(coordsOut['x'][1]))
yMin = int(round(coordsOut['y'][0]))
yMax = int(round(coordsOut['y'][1]))
print("xmin xmax ymin ymax", xMin, xMax, yMin, yMax)
print("xMax-xMin", xMax-xMin, "yMax-yMin", yMax-yMin)
if abs((xMax-xMin) - (yMax-yMin)) == 1:
    print('trying to fix it')
    if abs(xMax-xMin) > abs(yMax-yMin):
        print('case 1')
        if yMin == 0:
            yMax -= 1
            print('ymax-1')
        else:
            yMin -= 1
            print('ymin-1')

    else:
        print('case 2')
        if xMin == 0:
            xMax -= 1
            print('xMax - 1')
        else:
            xMin -= 1
            print('xmin-1')

print("xmin xmax ymin ymax", xMin, xMax, yMin, yMax)
print("xMax-xMin", xMax-xMin, "yMax-yMin", yMax-yMin)
assert xMax-xMin == yMax-yMin, "The selected FFT area does not appear to be square, make sure to hold shift when selecting the area of interest"
croppedCL = outAveraged[centerSliceIndex, yMin:yMax, xMin:xMax]
peakCoords = detect_peaks(croppedCL, 8).nonzero()

xPeakCoordsRaw = peakCoords[1] + xMin
yPeakCoordsRaw = peakCoords[0] + yMin
coordsToCheck = np.vstack((xPeakCoordsRaw, yPeakCoordsRaw)).T
validCoordsCheck = imageHandler.boundaryPoly.contains_points(coordsToCheck)
validCoordsCheck = np.vstack((validCoordsCheck, validCoordsCheck)).T
validCoordsRaw = np.ma.MaskedArray(coordsToCheck, mask=~validCoordsCheck).compressed()
validCoords = validCoordsRaw.reshape(int(len(validCoordsRaw)/2), 2)

# validCoords = [coordsToCheck[n,:] for n in range(len(coordsToCheck)) if validCoordsCheck[n]]
xPeakCoords = validCoords[:, 0] - xMin
yPeakCoords = validCoords[:, 1] - yMin


gaussianSigma = 1
windowSize = 4
truncateWindow = (((windowSize - 1)/2)-0.5)/gaussianSigma

croppedCL_Blurred = gaussian_filter(croppedCL, sigma=gaussianSigma, truncate=truncateWindow)
peakCoordsBlurred = detect_peaks(croppedCL_Blurred, 4).nonzero()
xPeakCoordsBlurredRaw = peakCoordsBlurred[1] + xMin
yPeakCoordsBlurredRaw = peakCoordsBlurred[0] + yMin

coordsToCheckBlurred = np.vstack((xPeakCoordsBlurredRaw, yPeakCoordsBlurredRaw)).T
validCoordsCheckBlurred = imageHandler.boundaryPoly.contains_points(coordsToCheckBlurred)
validCoordsCheckBlurred = np.vstack((validCoordsCheckBlurred, validCoordsCheckBlurred)).T
validCoordsRawBlurred = np.ma.MaskedArray(coordsToCheckBlurred, mask=~validCoordsCheckBlurred).compressed()
validCoordsBlurred = validCoordsRawBlurred.reshape(int(len(validCoordsRawBlurred)/2), 2)

# validCoords = [coordsToCheck[n,:] for n in range(len(coordsToCheck)) if validCoordsCheck[n]]
xPeakCoordsBlurred = validCoordsBlurred[:, 0] - xMin
yPeakCoordsBlurred = validCoordsBlurred[:, 1] - yMin

# peaks, idmap, promap, parentmap = getProminence(croppedCL, 0.2, min_area=None, include_edge=True)
print('here')
_, axs = plt.subplots(figsize=(8, 8), nrows=1, ncols=2)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
axs[0].imshow(croppedCL, interpolation='none', cmap='plasma', norm=LogNorm())
axs[0].scatter(xPeakCoords, yPeakCoords, marker='x')
axs[0].margins(x=0, y=0)

axs[1].imshow(croppedCL_Blurred, interpolation='none', cmap='plasma', norm=LogNorm())
axs[1].scatter(xPeakCoordsBlurred, yPeakCoordsBlurred, marker='x')
axs[1].margins(x=0, y=0)

axs[0].axis('equal')
axs[1].axis('equal')
plt.show()
plt.close()

# TODO: Not sure which to use, try integrating all of them

fftCroppedCL = abs(np.fft.fftshift(np.fft.fft2(croppedCL)))
fftLogCroppedCL = abs(np.fft.fftshift(np.fft.fft2(np.log10(croppedCL))))

centerFFTCoords = np.unravel_index(np.argmax(fftCroppedCL, axis=None), fftCroppedCL.shape)
radialProfile = radial_profile(fftCroppedCL, centerFFTCoords)
radialLogProfile = radial_profile(fftLogCroppedCL, centerFFTCoords)

_, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
plt.imshow(fftCroppedCL, interpolation='none', cmap='plasma')
ax.margins(x=0)
plt.axis('equal')
plt.show()

_, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
plt.imshow(fftCroppedCL, interpolation='none', cmap='plasma', norm=LogNorm())
ax.margins(x=0)
plt.axis('equal')
plt.show()

plt.plot(radialProfile)
plt.show()

#
# _, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# # TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
# plt.imshow(fftLogCroppedCL, interpolation='none', cmap='plasma')
# ax.margins(x=0)
# plt.axis('equal')
# plt.show()
#
#
# _, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# # TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
# plt.imshow(fftLogCroppedCL, interpolation='none', cmap='plasma', norm=LogNorm())
# ax.margins(x=0)
# plt.axis('equal')
# plt.show()
#
# plt.plot(radialLogProfile)
# plt.show()


