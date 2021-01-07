import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, Slider, PolygonSelector
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

from shapely.geometry import Point, LineString, MultiLineString, Polygon

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import gaussian_filter
from scipy.stats import lognorm

from Utility.peak_prominence2d import *

import joblib
import contextlib
import pickle
import multiprocessing
from tqdm import tqdm
num_cores = multiprocessing.cpu_count()


# User adjustable parameters
saveFigures = False
pixelScale = 49  # nm per pixel
backgroundPercentagePoint = 0.2  # the value that is higher than x percent of all valid points in the ROI
requiredBackgroundProminence = 7  # Required peak height as multiple of backgroundPercentagePoint to be a valid peak or number of standard deviations above background average

averagedSlices = 5  # Averaged wavelengths per center wavelength (symmetric)
gaussianSigma = 1
windowSize = 3
truncateWindow = (((windowSize - 1) / 2) - 0.5) / gaussianSigma
# rawCL = np.loadtxt('5min_Sample2.txt')
rawCL = np.loadtxt('Sample3_60min.txt')


# rawCL = np.loadtxt('CL Spectrum Image_12mW_3min_5_feb10.txt')

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def setAxisTicks(axisHandle, secondaryAxis=False):
    axisHandle.minorticks_on()
    if secondaryAxis:
        axisHandle.tick_params(which='both', axis='both', direction='in', top=True, bottom=False, left=False,
                               right=False)
    else:
        axisHandle.tick_params(which='both', axis='both', direction='in', top=False, bottom=True, left=True, right=True)
    axisHandle.tick_params(which='major', axis='both', direction='in', length=8, width=1)
    axisHandle.tick_params(which='minor', axis='both', direction='in', length=4, width=1)
    axisHandle.xaxis.set_minor_locator(AutoMinorLocator(2))
    axisHandle.yaxis.set_minor_locator(AutoMinorLocator(2))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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


def checkValidNeighbors(CLFrame, rowNum, colNum):
    if (CLFrame[rowNum - 1, colNum] > 1 and CLFrame[rowNum + 1, colNum] > 1 and
            CLFrame[rowNum - 1, colNum - 1] > 1 and CLFrame[rowNum - 1, colNum + 1] > 1 and
            CLFrame[rowNum + 1, colNum - 1] > 1 and CLFrame[rowNum + 1, colNum + 1] > 1 and
            CLFrame[rowNum, colNum - 1] > 1 and CLFrame[rowNum, colNum + 1] > 1):
        return True
    return False


def isPeak(CLFrame, rowNum, colNum):
    if (CLFrame[rowNum, colNum] > CLFrame[rowNum - 1, colNum] and CLFrame[rowNum, colNum] > CLFrame[rowNum + 1, colNum] and
            CLFrame[rowNum, colNum] > CLFrame[rowNum - 1, colNum - 1] and CLFrame[rowNum, colNum] > CLFrame[rowNum - 1, colNum + 1] and
            CLFrame[rowNum, colNum] > CLFrame[rowNum + 1, colNum - 1] and CLFrame[rowNum, colNum] > CLFrame[rowNum + 1, colNum + 1] and
            CLFrame[rowNum, colNum] > CLFrame[rowNum, colNum - 1] and CLFrame[rowNum, colNum] > CLFrame[rowNum, colNum + 1]):
        return True
    return False


def simplePeaks(CLFrame):
    xCoords = []
    yCoords = []
    for rowNum in range(np.shape(CLFrame)[0]):
        for colNum in range(np.shape(CLFrame)[1]):
            if CLFrame[rowNum, colNum] > 1:
                if rowNum > 0 and colNum > 0:
                    if checkValidNeighbors(CLFrame, rowNum, colNum):
                        if isPeak(CLFrame, rowNum, colNum):
                            xCoords.append(colNum)
                            yCoords.append(rowNum)
    return yCoords, xCoords


def getBackgroundPoints(CLFrame, backgroundPercentagePoint):
    # # Uses the x percentage lower point (ie 250th point if 1000 points) to define background level
    allYValues = np.ndarray.flatten(CLFrame)
    inBoundsYValues = allYValues[allYValues > 1]
    inBoundsYValues.sort()
    backgroundPoints = np.partition(inBoundsYValues, int(len(inBoundsYValues)*backgroundPercentagePoint))[:int(len(inBoundsYValues)*backgroundPercentagePoint)]
    return backgroundPoints


def findPeaks(CLFrame, pixelSize, backgroundPercentagePoint, requiredBackgroundProminence):
    # print(str(wavelengths[wavelengthNumber]) + "nm")
    # peakCoords = detect_peaks(CLFrame, 8).nonzero()  # Was set at 4 for the blurred one?
    peakCoords = simplePeaks(CLFrame)
    xPeakCoordsRaw = peakCoords[1]
    yPeakCoordsRaw = peakCoords[0]
    coordsToCheck = np.vstack((xPeakCoordsRaw, yPeakCoordsRaw)).T
    inBoundsCoordsCheck = imageHandler.boundaryPoly.contains_points(coordsToCheck)
    inBoundsCoordsCheck = np.vstack((inBoundsCoordsCheck, inBoundsCoordsCheck)).T
    inBoundsCoordsRaw = np.ma.MaskedArray(coordsToCheck, mask=~inBoundsCoordsCheck).compressed()
    inBoundsCoords = inBoundsCoordsRaw.reshape(int(len(inBoundsCoordsRaw)/2), 2)
    inBoundsCoordsX = inBoundsCoords[:, 0]
    inBoundsCoordsY = inBoundsCoords[:, 1]
    peakHeights = CLFrame[inBoundsCoordsY, inBoundsCoordsX]



    # # Uses the x percentage lower point (ie 250th point if 1000 points) to define background level

    backgroundPoints = getBackgroundPoints(CLFrame, backgroundPercentagePoint)
    # minimumValidYValue = requiredBackgroundProminence * backgroundPoints[int(len(inBoundsYValues)*backgroundPercentagePoint)]

    # Uses the average and standard deviation of the lowest x percentage of points (ie the lowest 250 points if 1000 points) to define background level
    # And requires peak to be above n standard deviations of those points
    backgroundAverage = np.mean(backgroundPoints)
    backgroundStdDev = np.std(backgroundPoints)
    minimumValidYValue = requiredBackgroundProminence * backgroundStdDev + backgroundAverage
    # print('Avg = ' + str(backgroundAverage) + ' StdDev = ' + str(backgroundStdDev) + 'MinimumYValue = ' + str(minimumValidYValue))
    # minimumValidYValue = 10

    validXCoords = pixelSize * inBoundsCoordsX[peakHeights > minimumValidYValue]
    validYCoords = pixelSize * inBoundsCoordsY[peakHeights > minimumValidYValue]
    return validXCoords, validYCoords


def getNearestNeighborDistances(xPeakCoords, yPeakCoords):
    NNdistances = []
    if len(xPeakCoords) > 1:
        for i in range(len(xPeakCoords)):
            rawNNdistances = []
            for j in range(len(yPeakCoords)):
                if i != j:
                    rawNNdistances.append(np.sqrt((xPeakCoords[i] - xPeakCoords[j]) ** 2 + (yPeakCoords[i] - yPeakCoords[j]) ** 2))
            rawNNdistances.sort()
            NNdistances.append(rawNNdistances)
    return NNdistances


def saveNNGraph(firstNNDict, pdfXValues, wavelength):
    shape, loc, scale = lognorm.fit(firstNNDict[wavelength], loc=0)
    pdfYValues = lognorm.pdf(pdfXValues, shape, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, dpi=300)
    plt.plot(firstNNDict[wavelength], np.zeros(len(firstNNDict[wavelength])), 'k|', markersize=30)
    plt.plot(pdfXValues, pdfYValues, 'k-')
    fig.canvas.set_window_title('First NN Distance - ' + str(532))
    plt.title('First NN Distances - ' + str(532))
    setAxisTicks(ax)
    ax.margins(x=0)
    plt.xlabel('Distance (nm)')
    plt.ylabel('Probability')
    # plt.show()
    fig.savefig('OutputFigures/'+str(wavelength)+'.png', bbox_inches='tight')
    plt.close(fig)


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
outAveragedBlurred = np.zeros((len(wavelengths), int(rawCL.shape[0]/len(wavelengths)), rawCL.shape[1]))
for centerSlice in range(len(wavelengths)):
    lowerSlice = max(0, centerSlice - int(((averagedSlices - 1) / 2)))
    upperSlice = min(len(wavelengths)-1, centerSlice + int(((averagedSlices - 1) / 2)))
    outAveraged[centerSlice, :, :] = np.mean(out[lowerSlice:upperSlice+1, :, :], 0)
    # outAveragedBlurred[centerSlice, :, :] = gaussian_filter(outAveraged[centerSlice, :, :], sigma=gaussianSigma, truncate=truncateWindow)
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

for centerSlice in range(len(wavelengths)):
    CLFrame = outAveraged[centerSlice, :, :]
    backgroundPoints = getBackgroundPoints(CLFrame, 0.01)
    backgroundAverage = np.mean(backgroundPoints)
    outAveragedBlurred[centerSlice, :, :] = gaussian_filter(CLFrame, sigma=gaussianSigma, truncate=truncateWindow) + backgroundAverage/2

outAveragedBlurred = outAveragedBlurred * imageHandler.imageMask + 1

# Looping Peak Finding
xPeakCoordsDict = {}
yPeakCoordsDict = {}
xPeakCoordsBlurredDict = {}
yPeakCoordsBlurredDict = {}
NNDict = {}
NNBlurredDict = {}
firstNNDict = {}
firstNNBlurredDict = {}
pdfXValues = list(range(0, 1001))
for wavelengthIndex in range(len(wavelengths)):
    wavelength = wavelengths[wavelengthIndex]

    xPeakCoordsDict[wavelength], yPeakCoordsDict[wavelength] = findPeaks(outAveraged[wavelengthIndex, :, :], pixelScale, backgroundPercentagePoint, requiredBackgroundProminence)
    xPeakCoordsBlurredDict[wavelength], yPeakCoordsBlurredDict[wavelength] = findPeaks(outAveragedBlurred[wavelengthIndex, :, :], pixelScale, backgroundPercentagePoint, requiredBackgroundProminence)

    # peaks, idmap, promap, parentmap = getProminence(CLFrame, 0.2, min_area=None, include_edge=True)
    NNDict[wavelength] = getNearestNeighborDistances(xPeakCoordsDict[wavelength], yPeakCoordsDict[wavelength])
    NNBlurredDict[wavelength] = getNearestNeighborDistances(xPeakCoordsBlurredDict[wavelength], yPeakCoordsBlurredDict[wavelength])

    # This is non blurred only right now
    firstNNDict[wavelength] = []
    for peak in NNDict[wavelength]:
        firstNNDict[wavelength].append(peak[0])


# Only does non-blurred right now
if saveFigures:
    with tqdm_joblib(tqdm(desc="Saving Nearest Neighbor Graphs", total=len(wavelengths))) as progress_bar:
        joblib.Parallel(n_jobs=num_cores)(joblib.delayed(saveNNGraph)(firstNNDict, pdfXValues, wavelength) for wavelength in wavelengths)

fig, axs = plt.subplots(figsize=(8, 8), nrows=1, ncols=2, sharex='all', sharey='all')
plt.subplots_adjust(bottom=0.18)
CLImage = axs[0].imshow(outAveraged[frameNum, :, :], interpolation='none', vmin=np.min(outAveraged[frameNum, :, :]), vmax=np.max(outAveraged[frameNum, :, :]), cmap='plasma', norm=LogNorm())
CLImagePeaks, = axs[0].plot(xPeakCoordsDict[wavelengths[frameNum]]/pixelScale, yPeakCoordsDict[wavelengths[frameNum]]/pixelScale, linestyle="", marker='x')
axs[0].margins(x=0, y=0)

CLImageBlurred = axs[1].imshow(outAveragedBlurred[frameNum, :, :], interpolation='none', vmin=np.min(outAveraged[frameNum, :, :]), vmax=np.max(outAveraged[frameNum, :, :]), cmap='plasma', norm=LogNorm())
CLImageBlurredPeaks, = axs[1].plot(xPeakCoordsBlurredDict[wavelengths[frameNum]]/pixelScale, yPeakCoordsBlurredDict[wavelengths[frameNum]]/pixelScale, linestyle="", marker='x')
manager = plt.get_current_fig_manager()
manager.window.maximize()
axs[1].margins(x=0, y=0)
axs[0].axis('equal')
axs[1].axis('equal')
axs[0].set_adjustable('box')
axs[1].set_adjustable('box')
axSlice = plt.axes([0.25, 0.1, 0.65, 0.03])
sSlice = Slider(axSlice, 'Wavelength (nm)', 0, len(wavelengths) - 1, valinit=int(len(wavelengths) / 2), valfmt='%0.0f')
sSlice.valtext.set_text(int(wavelengths[int(len(wavelengths) / 2)]))


def update(_):
    sSlice.valtext.set_text(int(wavelengths[int(sSlice.val)]))
    CLImage.set_data(outAveraged[int(sSlice.val), :, :])
    CLImage.vmin = np.min(outAveraged[int(sSlice.val), :, :])
    CLImage.vmax = np.max(outAveraged[int(sSlice.val), :, :])
    CLImagePeaks.set_data(xPeakCoordsDict[wavelengths[int(sSlice.val)]]/pixelScale, yPeakCoordsDict[wavelengths[int(sSlice.val)]]/pixelScale)

    CLImageBlurred.set_data(outAveragedBlurred[int(sSlice.val), :, :])
    CLImageBlurred.vmin = np.min(outAveragedBlurred[int(sSlice.val), :, :])
    CLImageBlurred.vmax = np.max(outAveragedBlurred[int(sSlice.val), :, :])
    CLImageBlurredPeaks.set_data(xPeakCoordsBlurredDict[wavelengths[int(sSlice.val)]]/pixelScale, yPeakCoordsBlurredDict[wavelengths[int(sSlice.val)]]/pixelScale)
    fig.canvas.draw_idle()


sSlice.on_changed(update)
plt.show()
plt.close()


# FFT Work
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


