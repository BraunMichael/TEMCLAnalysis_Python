import os
import re
import math
import json
import warnings
import jsonpickle
import numpy as np
import tkinter
from tkinter import Tk, filedialog
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, SpanSelector, Button
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from bisect import bisect_left

planck = 4.135667696 * (10 ** -15)  # eV * s
speedOfLight = 299792458  # m/s


class VirtualSlit:
    def __init__(self):
        self.width = 10
        self.integratedSpectra = None


class SpectrumData:
    def __init__(self, xVals: list = None, intensity: list = None, nakedFileName: str = ''):
        dataSortInd = xVals.argsort()
        self.xVals = xVals[dataSortInd]
        self.intensity = intensity[dataSortInd]
        self.nakedFileName = nakedFileName
        self.numXVals = len(self.xVals)
        self.minX = min(self.xVals)
        self.maxX = max(self.xVals)
        self.xRange = abs(self.maxX - self.minX)
        self.background = None
        self.numInterpolatedXVals = 10001
        self.interpolatedXVals = np.linspace(self.minX, self.maxX, num=self.numInterpolatedXVals, endpoint=True)
        # self.interpolationFunction = interp1d(self.xVals, savgol_filter(self.intensity, 9, 2), kind='cubic')
        self.interpolationFunction = interp1d(self.xVals, self.intensity, kind='cubic')

        self.interpolatedIntensity = self.interpolationFunction(self.interpolatedXVals)


class SetupOptions:
    def __init__(self):
        self.dataFilePath = ''
        self.isNormalized = False


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def wavelengthnmToEnergyEV(wavelength):
    if isinstance(wavelength, np.ndarray):
        wavelength[wavelength == 0] = 1e-15  # Don't try to divide by 0
    elif wavelength == 0:  # For now assuming it's otherwise just a single number
        wavelength = 1e-15
    return (planck * speedOfLight * 10 ** 9) / wavelength


def energyEVToWavelengthnm(energy):
    if isinstance(energy, np.ndarray):
        energy[energy == 0] = 1e-15  # Don't try to divide by 0
    elif energy == 0:  # For now assuming it's otherwise just a single number
        energy = 1e-15
    return (planck * speedOfLight * 10 ** 9) / energy


def closestNumAndIndex(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], pos
    if pos == len(myList):
        return myList[-1], pos
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after, pos
    else:
        return before, pos - 1


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getNakedNameFromFilePath(name):
    head, tail = os.path.split(name)
    nakedName, fileExtension = os.path.splitext(tail)
    return nakedName


def readDataFile(xrdFileNameFunc):
    xValsList = []
    intensityList = []
    delimiters = ' ', ',', ', ', '\t', '\n'
    regexPattern = '|'.join(map(re.escape, delimiters))

    with open(xrdFileNameFunc, 'r', encoding='ascii', errors='surrogateescape') as file:
        for line in file:
            splitLine = re.split(regexPattern, line)
            if all([is_number(splitLine[0]), is_number(splitLine[1])]):
                xValsList.append(float(splitLine[0]))
                intensityList.append(float(splitLine[1]))
    return np.asarray([xValsList, intensityList])


def calculateSingleAveragedWavelength(wavelengths, intensities, centerSlice, wavelengthWindow):
    _, proposedLowerSliceIndex = closestNumAndIndex(wavelengths, wavelengths[centerSlice] - 0.5 * wavelengthWindow)
    lowerSlice = max(0, proposedLowerSliceIndex)
    _, proposedUpperSliceIndex = closestNumAndIndex(wavelengths, wavelengths[centerSlice] + 0.5 * wavelengthWindow)
    upperSlice = min(len(wavelengths) - 1, proposedUpperSliceIndex)
    # simulatedSpectraPointIntensity = np.sum(intensities[lowerSlice:upperSlice + 1])
    simulatedSpectraPointIntensity = np.trapz(intensities[lowerSlice:upperSlice + 1], wavelengths[lowerSlice:upperSlice + 1])
    return simulatedSpectraPointIntensity


def simulatedSpectra(spectrumData: SpectrumData, wavelengthWindow):
    simulatedSpectraData = [calculateSingleAveragedWavelength(spectrumData.interpolatedXVals, spectrumData.interpolatedIntensity, centerSlice, wavelengthWindow) for centerSlice in range(spectrumData.numInterpolatedXVals)]
    return simulatedSpectraData


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


def getData(fileName):
    if not fileName:
        quit()
    nakedRawFileName = getNakedNameFromFilePath(fileName)
    print("Working on:", nakedRawFileName)
    rawData = readDataFile(fileName)
    return rawData, nakedRawFileName


def plotSetup(fig, ax, fileName: str, windowTitleSuffix: str, plotXLabel: str, plotYLabel: str,
              setupOptions: SetupOptions, withTopAxis: bool = False):
    fig.canvas.set_window_title(fileName + '_' + windowTitleSuffix)

    setAxisTicks(ax)
    if plotXLabel:
        ax.set_xlabel(plotXLabel)
    if plotYLabel:
        ax.set_ylabel(plotYLabel)

    if withTopAxis:
        secax = ax.secondary_xaxis('top', functions=(energyEVToWavelengthnm, wavelengthnmToEnergyEV))
        secax.set_xlabel('Wavelength (nm)')
        setAxisTicks(secax, True)


def SimulatedSpectraPlotting(spectrumData: SpectrumData, virtualSlit: VirtualSlit, setupOptions: SetupOptions):
    virtualSlit.width = 3
    simulatedSpectraData = simulatedSpectra(spectrumData, virtualSlit.width)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15)
    ax.margins(x=0)

    yVals = spectrumData.interpolatedIntensity
    if setupOptions.isNormalized:
        yVals = yVals/np.max(yVals)
    yLabel = 'Intensity'

    ax.plot(spectrumData.interpolatedXVals, yVals, 'k')
    simulated, = ax.plot(spectrumData.interpolatedXVals, simulatedSpectraData, 'b', label="Simulated")
    plt.legend(loc='best')
    # plotSetup(fig, ax, spectrumData.nakedFileName, 'SimulatedResolution', plotXLabel='Energy (eV)', plotYLabel=yLabel, setupOptions=setupOptions, withTopAxis=True)
    plotSetup(fig, ax, spectrumData.nakedFileName, 'SimulatedResolution', plotXLabel='Energy (eV)',
              plotYLabel=yLabel, setupOptions=setupOptions, withTopAxis=False)

    axWidth = plt.axes([0.25, 0.05, 0.65, 0.03])
    sWidth = Slider(axWidth, 'Resolution', 0.1, 200, valinit=virtualSlit.width)
    sWidth.valtext.set_text(virtualSlit.width)

    def update(_):
        sWidth.valtext.set_text('{:.1f}'.format(sWidth.val))
        simulatedSpectraDataUpdate = simulatedSpectra(spectrumData, sWidth.val)
        if setupOptions.isNormalized:
            simulated.set_ydata(simulatedSpectraDataUpdate/np.max(simulatedSpectraDataUpdate))
        else:
            simulated.set_ydata(simulatedSpectraDataUpdate)
        fig.canvas.draw_idle()

    sWidth.on_changed(update)

    plt.show(block=True)
    virtualSlit.width = sWidth.val
    plt.close()


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
    listName = getFileOrDirList('file', titleMessage, '.txt .xy .csv .dat',
                                entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def get_setupOptions():
    try:
        with open('SetupOptionsJSON_SpectralResolutionSimulation.txt') as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, dataFileEntryText, isNormalized):
    setupOptions.dataFilePath = dataFileEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.isNormalized = isNormalized.get()
    with open('SetupOptionsJSON_SpectralResolutionSimulation.txt', 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions):
    win.title("Spectrum Data Processing Setup UI")
    dataFileEntryText = tkinter.StringVar(value=setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~'))
    isNormalized = tkinter.BooleanVar(value=setupOptions.isNormalized)

    tkinter.Label(win, text="Data File:").grid(row=0, column=0)
    dataFileEntry = tkinter.Entry(win, textvariable=dataFileEntryText)
    dataFileEntry.grid(row=1, column=0)
    dataFileEntry.config(width=len(setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~')))
    dataFileButton = tkinter.Button(win, text='Choose File',
                                    command=lambda: get_file(dataFileEntry, dataFileEntryText, 'Choose Data File'))
    dataFileButton.grid(row=1, column=1)

    item_Label = tkinter.Label(win, text="Plot in Normalized Scale?", name='isNormalized_Label')
    item_Label.grid(row=2, column=0)
    r1isNormalized = tkinter.Radiobutton(win, text="Yes", variable=isNormalized, value=1, name='isNormalized_YesButton')
    r2isNormalized = tkinter.Radiobutton(win, text="No", variable=isNormalized, value=0, name='isNormalized_NoButton')
    r1isNormalized.grid(row=2, column=1)
    r2isNormalized.grid(row=2, column=2)

    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, dataFileEntryText, isNormalized))
    win.mainloop()


def main():
    setupOptions = get_setupOptions()  # Read previously used setupOptions
    uiInput(Tk(), setupOptions)  # UI to set configuration and get the input data files, takes the first 2 columns of a text, csv, dat, or xy file, string headers are ok and will be ignored
    rawData, nakedRawFileName = getData(setupOptions.dataFilePath)  # Read first 2 columns of a text, csv, dat, or xy file, string headers are ok and will be ignored
    # spectrumData = SpectrumData(wavelengthnmToEnergyEV(rawData[0]), rawData[1], nakedRawFileName)  # Make SpectrumData object and store data in it
    spectrumData = SpectrumData(rawData[0], rawData[1], nakedRawFileName)  # Make SpectrumData object and store data in it

    virtualSlit = VirtualSlit()  # Initialize RollingBall object
    SimulatedSpectraPlotting(spectrumData, virtualSlit, setupOptions)  # Interactive rolling ball background subtraction


if __name__ == "__main__":
    main()
