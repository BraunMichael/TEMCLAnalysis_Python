import os
import re
import math
import json
import warnings
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, Slider, PolygonSelector
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

from shapely.geometry import Point, LineString, MultiLineString, Polygon

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import gaussian_filter
from scipy.stats import lognorm

from Utility.peak_prominence2d import *
from Utility.AFMAnalyzeUI import *
from Utility.Utilities import *

import joblib
import contextlib
import pickle
import multiprocessing
from tqdm import tqdm
num_cores = multiprocessing.cpu_count()

scaleBarMicronsPerPixel = 0.001953125  #For 512 image with no borders

# User adjustable parameters
saveFigures = True
backgroundPercentagePoint = 0.2  # the value that is higher than x percent of all valid points in the ROI
requiredBackgroundProminence = 7  # Required peak height as multiple of backgroundPercentagePoint to be a valid peak or number of standard deviations above background average

averagedSlices = 5  # Averaged wavelengths per center wavelength (symmetric)
gaussianSigma = 1
windowSize = 3
truncateWindow = (((windowSize - 1) / 2) - 0.5) / gaussianSigma

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


def getNakedNameFromFilePath(name):
    head, tail = os.path.split(name)
    nakedName, fileExtension = os.path.splitext(tail)
    return nakedName


def getFileOrDirList(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = None,
                     initialDirOrFile: str = os.getcwd()):
    if os.path.isfile(initialDirOrFile) or os.path.isdir(initialDirOrFile):
        initialDir = os.path.split(initialDirOrFile)[0]
    else:
        initialDir = initialDirOrFile
    root = Tk()
    root.withdraw()
    assert fileOrFolder.lower() == 'file' or fileOrFolder.lower() == 'folder', "Only file or folder is an allowed string choice for fileOrFolder"
    if fileOrFolder.lower() == 'file':
        fileOrFolderList = filedialog.askopenfilename(initialdir=initialDir, title=titleStr,
                                                      filetypes=[(fileTypes + "file", fileTypes)])
    else:  # Must be folder from assert statement
        fileOrFolderList = filedialog.askdirectory(initialdir=initialDir, title=titleStr)
    if not fileOrFolderList:
        fileOrFolderList = initialDirOrFile
    root.destroy()
    return fileOrFolderList


def get_file(entryField, entryFieldText, titleMessage):
    listName = getFileOrDirList('folder', titleMessage, '.txt .xy .csv .dat',
                                entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def get_setupOptions():
    try:
        with open('SetupOptionsJSON_CLanalysis.txt') as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, dataFileEntryText, doAlignment):
    setupOptions.dataFilePath = dataFileEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.doAlignment = doAlignment.get()
    with open('SetupOptionsJSON_CLanalysis.txt', 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions):
    win.title("Spectrum Data Processing Setup UI")
    dataFileEntryText = tkinter.StringVar(value=setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~'))
    doAlignment = tkinter.BooleanVar(value=setupOptions.doAlignment)

    tkinter.Label(win, text="Data File:").grid(row=0, column=0)
    dataFileEntry = tkinter.Entry(win, textvariable=dataFileEntryText)
    dataFileEntry.grid(row=1, column=0)
    dataFileEntry.config(width=len(setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~')))
    dataFileButton = tkinter.Button(win, text='Choose File',
                                    command=lambda: get_file(dataFileEntry, dataFileEntryText, 'Choose Data File'))
    dataFileButton.grid(row=1, column=1)

    item_Label = tkinter.Label(win, text="Perform Image Stack Alignment?", name='doAlignment_Label')
    item_Label.grid(row=2, column=0)
    r1doAlignment = tkinter.Radiobutton(win, text="Yes", variable=doAlignment, value=1, name='doAlignment_YesButton')
    r2doAlignment = tkinter.Radiobutton(win, text="No", variable=doAlignment, value=0, name='doAlignment_NoButton')
    r1doAlignment.grid(row=2, column=1)
    r2doAlignment.grid(row=2, column=2)

    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, dataFileEntryText, doAlignment))
    win.mainloop()


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


def isPeak(AFMFrame, rowNum, colNum):
    if (AFMFrame[rowNum, colNum] > AFMFrame[rowNum - 1, colNum] and AFMFrame[rowNum, colNum] > AFMFrame[rowNum + 1, colNum] and
            AFMFrame[rowNum, colNum] > AFMFrame[rowNum - 1, colNum - 1] and AFMFrame[rowNum, colNum] > AFMFrame[rowNum - 1, colNum + 1] and
            AFMFrame[rowNum, colNum] > AFMFrame[rowNum + 1, colNum - 1] and AFMFrame[rowNum, colNum] > AFMFrame[rowNum + 1, colNum + 1] and
            AFMFrame[rowNum, colNum] > AFMFrame[rowNum, colNum - 1] and AFMFrame[rowNum, colNum] > AFMFrame[rowNum, colNum + 1]):
        return True
    return False


def simplePeaks(AFMFrame):
    xCoords = []
    yCoords = []
    for rowNum in range(np.shape(AFMFrame)[0] - 1):
        for colNum in range(np.shape(AFMFrame)[1] - 1):
            if AFMFrame[rowNum, colNum] > 1:
                if rowNum > 0 and colNum > 0:
                    if checkValidNeighbors(AFMFrame, rowNum, colNum):
                        if isPeak(AFMFrame, rowNum, colNum):
                            xCoords.append(colNum)
                            yCoords.append(rowNum)
    return yCoords, xCoords


def getBackgroundPoints(AFMFrame, backgroundPercentagePoint):
    # # Uses the x percentage lower point (ie 250th point if 1000 points) to define background level
    allYValues = np.ndarray.flatten(AFMFrame)
    inBoundsYValues = allYValues[allYValues > 1]
    inBoundsYValues.sort()
    backgroundPoints = np.partition(inBoundsYValues, int(len(inBoundsYValues)*backgroundPercentagePoint))[:int(len(inBoundsYValues)*backgroundPercentagePoint)]
    return backgroundPoints


def findPeaks(AFMFrame, pixelSize, backgroundPercentagePoint, requiredBackgroundProminence):
    # print(str(wavelengths[wavelengthNumber]) + "nm")
    # peakCoords = detect_peaks(CLFrame, 8).nonzero()  # Was set at 4 for the blurred one?
    peakCoords = simplePeaks(AFMFrame)
    xPeakCoordsRaw = peakCoords[1]
    yPeakCoordsRaw = peakCoords[0]
    peakHeights = AFMFrame[yPeakCoordsRaw, xPeakCoordsRaw]

    # # Uses the x percentage lower point (ie 250th point if 1000 points) to define background level

    backgroundPoints = getBackgroundPoints(AFMFrame, backgroundPercentagePoint)
    # minimumValidYValue = requiredBackgroundProminence * backgroundPoints[int(len(inBoundsYValues)*backgroundPercentagePoint)]

    # Uses the average and standard deviation of the lowest x percentage of points (ie the lowest 250 points if 1000 points) to define background level
    # And requires peak to be above n standard deviations of those points
    backgroundAverage = np.mean(backgroundPoints)
    backgroundStdDev = np.std(backgroundPoints)
    minimumValidYValue = requiredBackgroundProminence * backgroundStdDev + backgroundAverage
    # print('Avg = ' + str(backgroundAverage) + ' StdDev = ' + str(backgroundStdDev) + 'MinimumYValue = ' + str(minimumValidYValue))
    # minimumValidYValue = 10

    validXCoords = pixelSize * np.array(xPeakCoordsRaw)[peakHeights > minimumValidYValue]
    validYCoords = pixelSize * np.array(yPeakCoordsRaw)[peakHeights > minimumValidYValue]
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


def saveNNGraph(firstNearestNeighborsList, pdfXValues, suffix):
    if len(firstNearestNeighborsList) > 3:
        shape, loc, scale = lognorm.fit(firstNearestNeighborsList, floc=0, scale=170)
        pdfYValues = lognorm.pdf(pdfXValues, shape, loc=loc, scale=scale)

        fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1, dpi=300)
        plt.plot(firstNearestNeighborsList, np.zeros(len(firstNearestNeighborsList)), 'k|', markersize=30)
        plt.plot(pdfXValues, pdfYValues, 'k-')
        fig.canvas.set_window_title('First NN Distance')
        plt.title('First NN Distances')
        setAxisTicks(ax)
        ax.margins(x=0)
        plt.xlabel('Distance (nm)')
        plt.ylabel('Probability')
        # plt.show()
        fig.savefig(os.path.join('OutputFigures', suffix+'.png'), bbox_inches='tight')
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


setupOptions = setupOptionsUI()
inputFileNames = setupOptions.dataFilePath

if isinstance(inputFileNames, str):
    inputFileNames = [inputFileNames]
fileNames = []
for inputFileName in inputFileNames:
    if inputFileName.endswith(('jpg', '.jpeg')):
        fileNames.append(inputFileName)

    if inputFileName.endswith(('.tiff', '.tif')):
        print("attempting to convert tiff to png")
        imagePath = inputFileName

        fileTypeEnding = imagePath[imagePath.rfind('.'):]
        pngPath = inputFileName.replace(fileTypeEnding, '.png')
        # pngPath = os.path.join(dirpath, pngName)
        rawImage = Image.open(imagePath)
        npImage = ((np.array(rawImage) + 1) / 256) - 1
        visImage = Image.fromarray(np.uint8(npImage), mode='L')
        visImage.save(pngPath, 'PNG')
        fileNames.append(pngPath)
        # os.remove(imagePath)


for inputFileName in fileNames:
    rawImage = Image.open(inputFileName)
    AFMSourceImage = np.array(rawImage)
    nakedRawFileName = getNakedNameFromFilePath(inputFileName)
    rawImage.close()
    rawBlurred = np.zeros(AFMSourceImage.shape)

    AFMBlurredImage = gaussian_filter(AFMSourceImage, sigma=gaussianSigma, truncate=truncateWindow)

    AFMSourceImage = np.where(AFMSourceImage == 0, AFMSourceImage + 1, AFMSourceImage)
    AFMBlurredImage = np.where(AFMBlurredImage == 0, AFMBlurredImage + 1, AFMBlurredImage)
pixelScale = 1000 * setupOptions.scaleBarWidthMicrons / AFMSourceImage.shape[0]  # nm per pixel

# Peak Finding
pdfXValues = list(range(0, 101))
xPeakCoords, yPeakCoords = findPeaks(AFMSourceImage, pixelScale, backgroundPercentagePoint, requiredBackgroundProminence)
xPeakCoordsBlurred, yPeakCoordsBlurred = findPeaks(AFMBlurredImage, pixelScale, backgroundPercentagePoint, requiredBackgroundProminence)

# peaks, idmap, promap, parentmap = getProminence(CLFrame, 0.2, min_area=None, include_edge=True)
nearestNeighbors = getNearestNeighborDistances(xPeakCoords, yPeakCoords)
nearestNeighborsBlurred = getNearestNeighborDistances(xPeakCoordsBlurred, yPeakCoordsBlurred)

firstNearestNeighbors = []
firstNearestNeighborsBlurred = []

# Keep just the first nearest neighbor
for peak in nearestNeighbors:
    firstNearestNeighbors.append(peak[0])
for peak in nearestNeighborsBlurred:
    firstNearestNeighborsBlurred.append(peak[0])

if saveFigures:
    if not os.path.isdir('OutputFolder'):
        os.mkdir('OutputFolder')

    saveNNGraph(firstNearestNeighbors, pdfXValues, "Unfiltered")
    saveNNGraph(firstNearestNeighborsBlurred, pdfXValues, "GaussianFiltered")

fig, axs = plt.subplots(figsize=(8, 8), nrows=1, ncols=2, sharex='all', sharey='all')
plt.subplots_adjust(bottom=0.18)
AFMImage = axs[0].imshow(AFMSourceImage, interpolation='none', vmin=np.min(AFMSourceImage), vmax=np.max(AFMSourceImage), cmap='plasma', norm=LogNorm())
AFMImagePeaks, = axs[0].plot(xPeakCoords / pixelScale, yPeakCoords / pixelScale, linestyle="", marker='x')
axs[0].margins(x=0, y=0)

AFMImageBlurred = axs[1].imshow(AFMBlurredImage, interpolation='none', vmin=np.min(AFMBlurredImage), vmax=np.max(AFMBlurredImage), cmap='plasma', norm=LogNorm())
AFMImageBlurredPeaks, = axs[1].plot(xPeakCoordsBlurred / pixelScale, yPeakCoordsBlurred / pixelScale, linestyle="", marker='x')
manager = plt.get_current_fig_manager()
manager.window.maximize()
axs[1].margins(x=0, y=0)
axs[0].axis('equal')
axs[1].axis('equal')
axs[0].set_adjustable('box')
axs[1].set_adjustable('box')

plt.show()
plt.close()


# # FFT Work
#
# # TODO: Not sure which to use, try integrating all of them
#
# fftCroppedCL = abs(np.fft.fftshift(np.fft.fft2(croppedCL)))
# fftLogCroppedCL = abs(np.fft.fftshift(np.fft.fft2(np.log10(croppedCL))))
#
# centerFFTCoords = np.unravel_index(np.argmax(fftCroppedCL, axis=None), fftCroppedCL.shape)
# radialProfile = radial_profile(fftCroppedCL, centerFFTCoords)
# radialLogProfile = radial_profile(fftLogCroppedCL, centerFFTCoords)
#
# _, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# # TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
# plt.imshow(fftCroppedCL, interpolation='none', cmap='plasma')
# ax.margins(x=0)
# plt.axis('equal')
# plt.show()
#
# _, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# # TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
# plt.imshow(fftCroppedCL, interpolation='none', cmap='plasma', norm=LogNorm())
# ax.margins(x=0)
# plt.axis('equal')
# plt.show()
#
# plt.plot(radialProfile)
# plt.show()

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


