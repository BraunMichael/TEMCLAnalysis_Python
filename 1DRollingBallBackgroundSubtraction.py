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
from lmfit.models import GaussianModel, PseudoVoigtModel, VoigtModel, LorentzianModel
from bisect import bisect_left

planck = 4.135667696 * (10 ** -15)  # eV * s
speedOfLight = 299792458  # m/s


class RollingBall:
    def __init__(self):
        self.radius = 1
        self.ratio = 10
        self.minimumRadius = 0.1
        self.maximumRadius = 10
        self.minimumRatio = 0.01
        self.maximumRatio = 100

        assert self.minimumRadius > 0, "You have a negative minimum rolling ball radius, must be positive"
        assert self.maximumRadius > self.minimumRadius, "Your maximum rolling ball radius is smaller than your minimum rolling ball radius"
        assert self.minimumRadius <= self.radius <= self.maximumRadius, "Your rollingBallRadius is outside the range set by your minimum and maximum radius values"
        assert self.minimumRatio > 0, "You have a negative minimum rolling ball radius, must be positive"
        assert self.maximumRatio > self.minimumRatio, "Your maximum rolling ball radius is smaller than your minimum rolling ball radius"
        assert self.minimumRatio <= self.ratio <= self.maximumRatio, "Your rollingBallRadius is outside the range set by your minimum and maximum radius values"


class SpectrumData:
    def __init__(self, xVals: list = None, intensity: list = None, nakedFileName: str = ''):
        dataSortInd = xVals.argsort()
        self.xVals = xVals[dataSortInd]
        self.intensity = intensity[dataSortInd]
        self.lnIntensity = np.array(np.log(self.intensity))
        self.nakedFileName = nakedFileName
        self.numXVals = len(self.xVals)
        self.minX = min(self.xVals)
        self.maxX = max(self.xVals)
        self.xRange = abs(self.maxX - self.minX)
        self.background = None
        self.bgSubIntensity = None  # ln of intensity
        self.expBgSubIntensity = None  # Raw intensity


class SetupOptions:
    def __init__(self):
        self.dataFilePath = ''
        self.darkFilePath = ''
        self.isXRD = True
        self.doBackgroundSubtraction = True
        self.isGeSnPL = False
        self.modelType = ("gaussian", 0)


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


def calculateXRDSnContent(twoTheta):
    wavelength = 1.5405929  # Angstroms, Cu ka1
    aGe = 5.65791  # Angstroms
    aSn = 6.4892  # Angstroms
    dSample = (wavelength / (2 * np.sin(np.deg2rad(0.5 * twoTheta))))
    h = 3
    k = 3
    l = 3
    hkl = [h, k, l]
    h2k2l2 = np.sum(np.square(hkl))
    aSample = np.sqrt((dSample ** 2) * h2k2l2)
    snContentPercent = 100 * (aSample - aGe) / (aSn - aGe)
    return snContentPercent


def calculateXRDSnContent_Zach(doubletheta_GeSn):
    """
    Given the double angle of the Ge-Sn peak, return the Sn comp of the Ge-Sn layer
    """
    theta_GeSn = doubletheta_GeSn / 2
    wavelength = 1.5406  # Angstroms

    a_DC_Sn = 6.4892  # Angstroms
    d333_DC_Sn = a_DC_Sn / math.sqrt(27)

    doubletheta_Ge = 90.0571  # degrees
    d333_Ge = wavelength / 2 / math.sin(doubletheta_Ge / 2 * math.pi / 180)  # Angstroms
    d333_GeSn = wavelength / 2 / math.sin(theta_GeSn * math.pi / 180)

    x = (d333_GeSn - d333_Ge) / (d333_DC_Sn - d333_Ge)
    return x * 100


def calculateTwoTheta(snContentPercent=0):
    wavelength = 1.5405929  # Angstroms, Cu ka1
    aGe = 5.65791  # Angstroms
    aSn = 6.4892  # Angstroms
    h = 3
    k = 3
    l = 3
    hkl = [h, k, l]
    h2k2l2 = np.sum(np.square(hkl))
    aSample = aGe + ((snContentPercent / 100) * (aSn - aGe))
    twoTheta = 2 * np.rad2deg(np.arcsin(0.5 * (wavelength / np.sqrt(aSample ** 2 / h2k2l2))))
    return twoTheta


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


def SnContent_To_IndirectConductionBandEnergy(snContent):
    a_param = 0.71075
    a_param_uncertainty = 0.00012944
    b_param = -0.74189
    b_param_uncertainty = 0.00416
    c_param = 0.02276
    c_param_uncertainty = 0.02697
    return a_param + b_param * snContent + c_param * snContent * snContent


def SnContent_To_ValenceBandEnergy(snContent):
    a_param = 0.08927
    a_param_uncertainty = 0.000275186
    b_param = 1.21497
    b_param_uncertainty = 0.00845
    c_param = -0.98709
    c_param_uncertainty = 0.05438
    return a_param + b_param * snContent + c_param * snContent * snContent


def SnContent_To_DirectConductionBandEnergy(snContent):
    a_param = 0.79704
    a_param_uncertainty = 0.00039548
    b_param = -1.63657
    b_param_uncertainty = 0.01182
    c_param = 0.46315
    c_param_uncertainty = 0.07498
    return a_param + b_param * snContent + c_param * snContent * snContent


def SnContent_To_DirectBandgap(snContent):
    return SnContent_To_DirectConductionBandEnergy(snContent) - SnContent_To_ValenceBandEnergy(snContent)


def SnContent_To_IndirectBandgap(snContent):
    return SnContent_To_IndirectConductionBandEnergy(snContent) - SnContent_To_ValenceBandEnergy(snContent)


def SnContent_To_LowestBandgap(snContent):
    if SnContent_To_DirectBandgap(snContent) < SnContent_To_IndirectBandgap(snContent):
        return SnContent_To_DirectBandgap(snContent)
    return SnContent_To_IndirectBandgap(snContent)


def IndirectBandgap_To_SnContent(energy):
    a_param = 0.29069
    a_param_uncertainty = 0.0000544
    b_param = -0.48176
    b_param_uncertainty = 0.00038
    c_param = 0.1022
    c_param_uncertainty = 0.000544
    return a_param + b_param * energy + c_param * energy * energy


def DirectBandgap_To_SnContent(energy):
    a_param = 0.3915
    a_param_uncertainty = 0.0001498
    b_param = -0.76635
    b_param_uncertainty = 0.0008976
    c_param = 0.22192
    c_param_uncertainty = 0.0012
    return a_param + b_param * energy + c_param * energy * energy


def Bandgap_To_LowestSnContent(energy):
    if DirectBandgap_To_SnContent(energy) < IndirectBandgap_To_SnContent(energy):
        return DirectBandgap_To_SnContent(energy)
    return IndirectBandgap_To_SnContent(energy)


def closestNumAndIndex(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
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


def calculateSingleBackground(measuredAngles, lnIntensities, angleNum, radiussquared, rollingBallRatioVal):
    centerAngle = measuredAngles[angleNum]
    centerIntensity = lnIntensities[angleNum]
    ballPoints = centerIntensity + rollingBallRatioVal * np.sqrt(
        radiussquared - ((measuredAngles - centerAngle) * (measuredAngles - centerAngle)))
    backgroundDifference = ballPoints - lnIntensities
    backgroundOffset = np.nanmax(backgroundDifference)
    return ballPoints - backgroundOffset


def rollingBallBackground(spectrumData: SpectrumData, rollingBallRatioVal, radius):
    radiussquared = radius * radius
    allBackgrounds = [calculateSingleBackground(spectrumData.xVals, spectrumData.lnIntensity, angleNum, radiussquared,
                                                rollingBallRatioVal) for angleNum in range(spectrumData.numXVals)]
    return np.nanmax(allBackgrounds, axis=0)


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
    if setupOptions.isXRD:
        if withTopAxis:
            secax = ax.secondary_xaxis('top', functions=(calculateXRDSnContent, calculateTwoTheta))
            secax.set_xlabel('Sn Content (%)')
            setAxisTicks(secax, True)
    else:  # isPL
        if withTopAxis:
            secax = ax.secondary_xaxis('top', functions=(energyEVToWavelengthnm, wavelengthnmToEnergyEV))
            secax.set_xlabel('Wavelength (nm)')
            setAxisTicks(secax, True)
            if setupOptions.isGeSnPL:
                pass
                # TODO: This doesn't really work right now, see comments below assertion statement
                # fig.subplots_adjust(top=0.8)
                #
                # axGeSn = ax.twiny()
                # axGeSn.xaxis.set_ticks_position("top")  # may not need this
                # axGeSn.xaxis.set_label_position("top")  # may not need this
                # axGeSn.spines["top"].set_position(("axes", 1.15))
                # axGeSn.set_frame_on(True)
                # axGeSn.patch.set_visible(False)
                # for sp in axGeSn.spines.values():
                #     sp.set_visible(False)
                # axGeSn.spines["top"].set_visible(True)
                # axGeSn.set_xlabel('Sn Content (%)')
                # axXmin, axXmax = ax.get_xlim()
                # assert axXmax < 1.5, "The data you are trying to plot is probably no GeSn PL/CL as your maximum energy ({0} eV) is > 1.5 eV".format(round(axXmax, 3))
                # A possible option: https://matplotlib.org/3.1.0/gallery/scales/custom_scale.html
                # Related: https://stackoverflow.com/questions/14845350/multiple-x-axis-which-are-nonlinear-to-each-other
                # And: https://matplotlib.org/3.1.0/gallery/scales/scales.html
                # This is also basically broken, scaling changes based on starting x axis range...
                # axGeSn.set_xlim(DirectBandgap_To_SnContent(axXmin), DirectBandgap_To_SnContent(axXmax))
                # Doesn't work, has issues with non-monotonic functions, but is probably better if we can fix it...
                # axGeSn.set_xscale('function', functions=(DirectBandgap_To_SnContent, SnContent_To_DirectBandgap))


def backgroundSubtractionPlotting(spectrumData: SpectrumData, rollingBall: RollingBall, setupOptions: SetupOptions):
    isXRD = setupOptions.isXRD
    rollingBallBackgroundData = rollingBallBackground(spectrumData, rollingBall.ratio, rollingBall.radius)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    ax.margins(x=0)
    ax.plot(spectrumData.xVals, spectrumData.lnIntensity, 'k')
    background, = ax.plot(spectrumData.xVals, rollingBallBackgroundData, 'r--', label="Rolling Ball Background")
    subtracted, = ax.plot(spectrumData.xVals, spectrumData.lnIntensity - rollingBallBackgroundData, 'b',
                          label="Background Subtracted")
    plt.legend(loc='best')
    if isXRD:
        plotSetup(fig, ax, spectrumData.nakedFileName, 'BackgroundSubtraction', plotXLabel='$2\\theta$',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)
        rollingBall.radius = spectrumData.xRange
    else:  # isPL
        plotSetup(fig, ax, spectrumData.nakedFileName, 'BackgroundSubtraction', plotXLabel='Energy (eV)',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)
        rollingBall.radius = spectrumData.xRange * 10
        rollingBall.ratio = 1

    axRadius = plt.axes([0.25, 0.05, 0.65, 0.03])
    axRatio = plt.axes([0.25, 0.10, 0.65, 0.03])

    log10xRangeBasis = np.log10(spectrumData.xRange / 4)
    if np.floor(log10xRangeBasis) == round(log10xRangeBasis):
        rollingBall.minimumRadius = 10 ** (np.floor(log10xRangeBasis) - 1)
        rollingBall.maximumRadius = 10 ** (np.ceil(log10xRangeBasis) + 1)
    else:
        rollingBall.minimumRadius = 10 ** np.floor(log10xRangeBasis)
        rollingBall.maximumRadius = 10 ** (np.ceil(log10xRangeBasis) + 2)
    sRadius = Slider(axRadius, 'Rolling Ball Radius', np.log10(rollingBall.minimumRadius), np.log10(rollingBall.maximumRadius), valinit=np.log10(rollingBall.radius))
    sRatio = Slider(axRatio, 'Aspect Ratio', np.log10(rollingBall.minimumRatio), np.log10(rollingBall.maximumRatio), valinit=np.log10(rollingBall.ratio))
    sRadius.valtext.set_text(rollingBall.radius)
    sRatio.valtext.set_text(rollingBall.ratio)

    def update(val):
        sRadius.valtext.set_text('{:.1f}'.format(10 ** sRadius.val))
        sRatio.valtext.set_text('{:.1f}'.format(10 ** sRatio.val))
        rollingBallBackgroundDataUpdate = rollingBallBackground(spectrumData, 10 ** sRatio.val, 10 ** sRadius.val)
        background.set_ydata(rollingBallBackgroundDataUpdate)
        subtracted.set_ydata(spectrumData.lnIntensity - rollingBallBackgroundDataUpdate)
        fig.canvas.draw_idle()

    sRadius.on_changed(update)
    sRatio.on_changed(update)

    plt.show(block=True)
    rollingBall.radius = 10 ** sRadius.val
    rollingBall.ratio = 10 ** sRatio.val
    plt.close()


def fittingRegionSelectionPlotting(spectrumData: SpectrumData, setupOptions: SetupOptions):
    # Each "region" limits the peak center position
    # Each MultiFit Area uses all the data in the selected area and tries to fit the regions within to the whole data,
    # Except the peak center of each region in the MultiFit area is limited to its selected region
    fig, ax = plt.subplots(figsize=(10, 8))
    if setupOptions.isXRD:
        plotSetup(fig, ax, spectrumData.nakedFileName, 'PeakFitting', plotXLabel='$2\\theta$',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)
    else:  # isPL
        plotSetup(fig, ax, spectrumData.nakedFileName, 'PeakFitting', plotXLabel='Energy (eV)',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)

    plt.subplots_adjust(bottom=0.2)
    ax.plot(spectrumData.xVals, spectrumData.bgSubIntensity, 'b')

    class RangeSelect:
        def __init__(self):
            self.coords = {}

        def __call__(self, xmin, xmax):
            indmin, indmax = np.searchsorted(spectrumData.xVals, (xmin, xmax))
            indmax = min(len(spectrumData.xVals) - 1, indmax)

            thisx = spectrumData.xVals[indmin:indmax]
            thisy = spectrumData.bgSubIntensity[indmin:indmax]
            self.coords['x'] = thisx
            self.coords['y'] = thisy
            (ymin, ymax) = ax.get_ylim()
            if len(ax.patches) > numRanges:
                ax.patches[-1].remove()
            if len(thisx) == 0:
                thisx = [0]
            rect = patches.Rectangle((min(thisx), ymin), max(thisx) - min(thisx), ymax - ymin, linewidth=1,
                                     edgecolor='none', facecolor='red', alpha=0.5, fill=True)
            ax.add_patch(rect)
            fig.canvas.draw()

    plt.show(block=False)
    rangeselect = RangeSelect()
    coordsList = []
    multiRegionCoordsList = []

    class Index(object):
        index = 1

        def addRegion(self, event):
            self.index += 1
            numRanges = self.index
            coordsList.append(rangeselect.coords.copy())
            ax.patches[-1].set_facecolor('green')
            span = SpanSelector(ax, rangeselect, 'horizontal', useblit=False,
                                rectprops=dict(alpha=0.3, facecolor='red'))
            plt.draw()

        def addMultiFitRegion(self, event):
            self.index += 1
            numRanges = self.index
            multiRegionCoordsList.append(rangeselect.coords.copy())
            ax.patches[-1].set_facecolor('blue')
            ax.patches[-1].set_alpha(0.2)
            span = SpanSelector(ax, rangeselect, 'horizontal', useblit=False,
                                rectprops=dict(alpha=0.3, facecolor='red'))
            plt.draw()

    callback = Index()
    axAddRegion = plt.axes([0.7, 0.02, 0.2, 0.075])
    bAdd = Button(axAddRegion, 'Add Region')
    axAddRegion = plt.axes([0.45, 0.02, 0.2, 0.075])
    bMulti = Button(axAddRegion, 'MultiFit Area')
    numRanges = 1
    span = SpanSelector(ax, rangeselect, 'horizontal', useblit=False, rectprops=dict(alpha=0.3, facecolor='red'))
    bAdd.on_clicked(callback.addRegion)
    bMulti.on_clicked(callback.addMultiFitRegion)

    plt.show(block=True)
    plt.close()
    assert len(coordsList) > 0, "There are no selected ranges in the coordsList"
    for multiIndex, multiFitRegion in enumerate(multiRegionCoordsList):
        multiFitRegionSet = set(multiFitRegion['x'])
        for subMultiIndex, subMultiFitRegion in enumerate(multiRegionCoordsList):
            if multiIndex != subMultiIndex:
                assert multiFitRegionSet.isdisjoint(subMultiFitRegion['x']), "You have selected overlapping MultiFit areas, this is not allowed."
    return coordsList, multiRegionCoordsList


def prepareFittingModels(roiCoordsList, modelType):
    modelList = []
    paramList = []
    index = 1
    for region in roiCoordsList:
        individualModelsList = []
        individualParamsList = []
        if isinstance(region, dict):
            # If the region is just a single region, make it a list so the for loops pulls a dict rather than a dict entry
            region = [region]
        for entry in region:
            prefixName = 'v' + str(index) + '_'
            index += 1
            # pull info out of region dict
            selectedXVals = entry['x']
            selectedYVals = np.exp(entry['y'])

            if modelType.lower() == 'voigt':
                mod = VoigtModel(prefix=prefixName)
            elif modelType.lower() == 'psuedovoigt':
                mod = PseudoVoigtModel(prefix=prefixName)
            elif modelType.lower() == 'lorentzian':
                mod = LorentzianModel(prefix=prefixName)
            elif modelType.lower() == 'gaussian':
                mod = GaussianModel(prefix=prefixName)
            else:
                assert True, "Entered model type is not supported"

            individualModelsList.append(mod)
            pars = mod.guess(selectedYVals, x=selectedXVals, negative=False)
            pars[prefixName + 'center'].set(min=min(selectedXVals), max=max(selectedXVals))
            pars[prefixName + 'amplitude'].set(min=0)
            pars[prefixName + 'sigma'].set(min=0)
            if modelType.lower() == 'voigt':
                pars[prefixName + 'gamma'].set(value=0.3, vary=True, expr='', min=0)
            individualParamsList.append(pars)
        combinedModel = individualModelsList[0]
        combinedParams = individualParamsList[0]
        if len(individualModelsList) > 1:
            for model, params in zip(individualModelsList[1:], individualParamsList[1:]):
                combinedModel += model
                combinedParams += params
        modelList.append(combinedModel)
        paramList.append(combinedParams)
    return modelList, paramList


def splitMultiFitModels(roiCoordsList, multiRegionCoordsList, modelType: str):
    combinedModelsList = []  # Each element of this list should be 1 combined model, each element is a list, if len == 1 then it's fit independently
    coordsList = []  # Defines the area to fit each element of combinedModelsList, see fittingRegionSelectionPlotting for more info
    if multiRegionCoordsList:
        for multiRegion in multiRegionCoordsList:
            multiRegionXValsSet = set(multiRegion['x'])
            combinedRegionList = []
            indicesToDelete = []
            for roiIndex, roi in enumerate(roiCoordsList):
                if not multiRegionXValsSet.isdisjoint(roi['x']):
                    # There is at least 1 element in common between the roi and multiregion
                    combinedRegionList.append(roi)
                    indicesToDelete.append(roiIndex)

            assert combinedRegionList, "There were no detected regions in the MultiFit Area"
            combinedModelsList.append(combinedRegionList)
            coordsList.append(multiRegion)
            # Be careful not to mess up indices of list while trying to delete based on index!
            for index in sorted(indicesToDelete, reverse=True):
                del (roiCoordsList[index])
        combinedModelsList.extend(
            roiCoordsList)  # After removing any entries in a MultiFit Area, add the rest of them onto the model list
        coordsList.extend(roiCoordsList)
    else:
        combinedModelsList = roiCoordsList
        coordsList = roiCoordsList
    return prepareFittingModels(combinedModelsList, modelType), coordsList


def xrdCalculationProcessing(spectrumData, centerXValsList, heightList, axs):
    proposedUserSubstrateTwoTheta = centerXValsList[heightList.index(max(heightList))]
    substrateModel = VoigtModel()
    params = substrateModel.guess(spectrumData.expBgSubIntensity, x=spectrumData.xVals, negative=False)
    out = substrateModel.fit(spectrumData.expBgSubIntensity, params, x=spectrumData.xVals)
    fullModelSubstrateTwoTheta = out.best_values['center']
    if abs(fullModelSubstrateTwoTheta - proposedUserSubstrateTwoTheta) <= 0.1:
        # looks like the user selected the substrate as a peak, use their value
        substrateTwoTheta = proposedUserSubstrateTwoTheta
    else:
        # Looks like the user did not select the substrate as a peak, use a global value from fitting all data
        substrateTwoTheta = fullModelSubstrateTwoTheta

    literatureSubstrateTwoTheta = calculateTwoTheta(snContentPercent=0)  # Reusing Sn content to 2theta equation
    twoThetaOffset = substrateTwoTheta - literatureSubstrateTwoTheta
    offsetCorrectedCenterTwoThetaList = np.asarray(centerXValsList) - twoThetaOffset
    for centerTwoTheta in offsetCorrectedCenterTwoThetaList:
        michaelSnContent = round(calculateXRDSnContent(centerTwoTheta), 1)
        print("Michael Comp:", michaelSnContent)
        print("Zach Comp:", round(calculateXRDSnContent_Zach(centerTwoTheta), 1))
        if abs(centerTwoTheta - literatureSubstrateTwoTheta) > 0.05:  # Don't draw one for the substrate
            _, centerIndex = closestNumAndIndex(spectrumData.xVals, centerTwoTheta + twoThetaOffset)
            an0 = axs[0].annotate(str(abs(michaelSnContent)),
                                  xy=(centerTwoTheta + twoThetaOffset, spectrumData.lnIntensity[centerIndex]),
                                  xycoords='data', xytext=(0, 72), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->", shrinkA=10, shrinkB=5, patchA=None,
                                                  patchB=None))
            an0.draggable()
            an1 = axs[1].annotate(str(abs(michaelSnContent)), xy=(
                centerTwoTheta + twoThetaOffset, spectrumData.bgSubIntensity[centerIndex]), xycoords='data',
                                  xytext=(0, 72), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->", shrinkA=10, shrinkB=5, patchA=None,
                                                  patchB=None))
            an1.draggable()


def plCalculationProcessing(spectrumData, centerXValsList, axs, isGeSnPL):
    if isGeSnPL:
        for centerWavelength in np.asarray(centerXValsList):
            # TODO: Update this with new equations
            snContent = round(calculateEnergyEVToSnContent(centerWavelength), 1)
            print("Sn Composition:", snContent)
            _, centerIndex = closestNumAndIndex(spectrumData.xVals, centerWavelength)
            an0 = axs[0].annotate(str(abs(snContent)),
                                  xy=(centerWavelength, spectrumData.lnIntensity[centerIndex]),
                                  xycoords='data', xytext=(0, 72), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->", shrinkA=10, shrinkB=5, patchA=None,
                                                  patchB=None))
            an0.draggable()
            an1 = axs[1].annotate(str(abs(snContent)),
                                  xy=(centerWavelength, spectrumData.bgSubIntensity[centerIndex]),
                                  xycoords='data', xytext=(0, 72), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->", shrinkA=10, shrinkB=5, patchA=None,
                                                  patchB=None))
            an1.draggable()


def snContentFittingPlotting(spectrumData: SpectrumData, roiCoordsList: list, multiRegionCoordsList: list,
                             setupOptions: SetupOptions):
    # TODO: Maybe implement this if isGeSnPL from UI, as a 2nd top axis to show wavelength, energy, and Sn content on the same axes
    #  like https://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
    #  And possibly better: https://stackoverflow.com/questions/25159495/multiple-y-axis-conversion-scales
    isXRD = setupOptions.isXRD
    (modelList, paramList), fittingCoordsList = splitMultiFitModels(roiCoordsList, multiRegionCoordsList,
                                                                    setupOptions.modelType[0])
    fig, axs = plt.subplots(ncols=2, figsize=(10, 8), gridspec_kw={'wspace': 0})
    if isXRD:
        plotSetup(fig, axs[0], spectrumData.nakedFileName, 'FittingResults', plotXLabel='$2\\theta$',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)
        plotSetup(fig, axs[1], spectrumData.nakedFileName, 'FittingResults', plotXLabel='$2\\theta$', plotYLabel='',
                  setupOptions=setupOptions, withTopAxis=True)
    else:  # isPL
        plotSetup(fig, axs[0], spectrumData.nakedFileName, 'FittingResults', plotXLabel='Energy (eV)',
                  plotYLabel='ln(Intensity)', setupOptions=setupOptions, withTopAxis=True)
        plotSetup(fig, axs[1], spectrumData.nakedFileName, 'FittingResults', plotXLabel='Energy (eV)', plotYLabel='',
                  setupOptions=setupOptions, withTopAxis=True)
    axs[0].plot(spectrumData.xVals, spectrumData.lnIntensity, 'k')
    rawYmin, _ = axs[0].get_ylim()
    axs[1].plot(spectrumData.xVals, spectrumData.bgSubIntensity, 'b')
    bgYmin, _ = axs[1].get_ylim()

    centerXValsList = []
    heightList = []
    for model, params, subCoords in zip(modelList, paramList, fittingCoordsList):
        out = model.fit(np.exp(subCoords['y']), params, x=subCoords['x'])
        print("Minimum center:", min(subCoords['x']), "Maximum center:", max(subCoords['x']))
        print(out.fit_report(min_correl=0.25))
        comps = out.eval_components(x=subCoords['x'])
        if len(comps) > 1:
            for _, component in comps.items():
                axs[1].plot(subCoords['x'], np.log(component), 'g--')
        axs[1].plot(subCoords['x'], np.log(out.best_fit), 'r--')
        for key, value in out.best_values.items():
            if 'center' in key:
                centerXValsList.append(value)
        for key in out.params.keys():
            if 'height' in key:
                heightList.append(out.params[key].value)

    if isXRD:
        xrdCalculationProcessing(spectrumData, centerXValsList, heightList, axs)
    else:  # isPL
        plCalculationProcessing(spectrumData, centerXValsList, axs, setupOptions.isGeSnPL)

    print("Results from:", spectrumData.nakedFileName)
    axs[0].set_ylim(bottom=rawYmin)
    axs[1].set_ylim(bottom=bgYmin)

    plt.show(block=True)
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


def hide_GeSnPL(win):
    if 'geSnPL_Label' in win.children:
        win.children['geSnPL_Label'].destroy()
        win.children['geSnPL_YesButton'].destroy()
        win.children['geSnPL_NoButton'].destroy()


def show_GeSnPL(win, isGeSnPL):
    if 'geSnPL_Label' not in win.children:
        item_Label = tkinter.Label(win, text="[PL Only] GeSn specific calculations?", name='geSnPL_Label')
        item_Label.grid(row=7, column=0)
        r1isGeSnPL = tkinter.Radiobutton(win, text="Yes", variable=isGeSnPL, value=1, name='geSnPL_YesButton')
        r2isGeSnPL = tkinter.Radiobutton(win, text="No", variable=isGeSnPL, value=0, name='geSnPL_NoButton')
        r1isGeSnPL.grid(row=7, column=1)
        r2isGeSnPL.grid(row=7, column=2)


def get_setupOptions():
    try:
        with open('SetupOptionsJSON.txt') as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, dataFileEntryText, darkFileEntryText, isXRD, doBackgroundSubtraction, isGeSnPL,
               modelType):
    setupOptions.dataFilePath = dataFileEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.darkFilePath = darkFileEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.isXRD = isXRD.get()
    setupOptions.doBackgroundSubtraction = doBackgroundSubtraction.get()
    setupOptions.isGeSnPL = isGeSnPL.get()
    setupOptions.modelType = modelType
    with open('SetupOptionsJSON.txt', 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions):
    win.title("Spectrum Data Processing Setup UI")
    dataFileEntryText = tkinter.StringVar(value=setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~'))
    darkFileEntryText = tkinter.StringVar(value=setupOptions.darkFilePath.replace(os.path.expanduser('~'), '~'))

    isXRD = tkinter.BooleanVar(value=setupOptions.isXRD)
    doBackgroundSubtraction = tkinter.BooleanVar(value=setupOptions.doBackgroundSubtraction)
    isGeSnPL = tkinter.BooleanVar(value=setupOptions.isGeSnPL)

    tkinter.Label(win, text="Data File:").grid(row=0, column=0)
    dataFileEntry = tkinter.Entry(win, textvariable=dataFileEntryText)
    dataFileEntry.grid(row=1, column=0)
    dataFileEntry.config(width=len(setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~')))
    dataFileButton = tkinter.Button(win, text='Choose File',
                                    command=lambda: get_file(dataFileEntry, dataFileEntryText, 'Choose Data File'))
    dataFileButton.grid(row=1, column=1)

    tkinter.Label(win, text="Dark File:").grid(row=2, column=0)
    darkFileEntry = tkinter.Entry(win, textvariable=darkFileEntryText)
    darkFileEntry.grid(row=3, column=0)
    dataFileEntry.config(width=len(setupOptions.darkFilePath.replace(os.path.expanduser('~'), '~')))
    darkFileButton = tkinter.Button(win, text='Choose File',
                                    command=lambda: get_file(darkFileEntry, darkFileEntryText, 'Choose Dark Scan File'))
    darkFileButton.grid(row=3, column=1)

    item_Label = tkinter.Label(win, text="XRD or PL/CL")
    item_Label.grid(row=4, column=0)
    r1isXRD = tkinter.Radiobutton(win, text="XRD", variable=isXRD, value=1, command=lambda: hide_GeSnPL(win))
    r2isXRD = tkinter.Radiobutton(win, text="PL/CL", variable=isXRD, value=0,
                                  command=lambda: show_GeSnPL(win, isGeSnPL))
    r1isXRD.grid(row=4, column=1)
    r2isXRD.grid(row=4, column=2)

    item_Label = tkinter.Label(win, text="Background subtraction (interactive)?")
    item_Label.grid(row=5, column=0)
    r1doBgSub = tkinter.Radiobutton(win, text="Yes", variable=doBackgroundSubtraction, value=1)
    r2doBgSub = tkinter.Radiobutton(win, text="No", variable=doBackgroundSubtraction, value=0)
    r1doBgSub.grid(row=5, column=1)
    r2doBgSub.grid(row=5, column=2)

    modelTypes = [("Gaussian", 0), ("Voigt", 1), ("Psuedovoigt", 2), ("Lorentzian", 3)]

    varModelType = tkinter.IntVar()
    varModelType.set(setupOptions.modelType[1])

    item_Label = tkinter.Label(win, text="Fitting Model")
    item_Label.grid(row=6, column=0)
    for model, number in modelTypes:
        radioModelType = tkinter.Radiobutton(win, text=model, variable=varModelType, value=number)
        radioModelType.grid(row=6, column=(number + 1))

    if setupOptions.isXRD:
        hide_GeSnPL(win)
    else:
        show_GeSnPL(win, isGeSnPL)
    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, dataFileEntryText, darkFileEntryText, isXRD,
                                                        doBackgroundSubtraction, isGeSnPL,
                                                        modelTypes[varModelType.get()]))
    win.mainloop()


def main():
    setupOptions = get_setupOptions()  # Read previously used setupOptions
    uiInput(Tk(), setupOptions)  # UI to set configuration and get the input data files, takes the first 2 columns of a text, csv, dat, or xy file, string headers are ok and will be ignored
    rawData, nakedRawFileName = getData(setupOptions.dataFilePath)  # Read first 2 columns of a text, csv, dat, or xy file, string headers are ok and will be ignored
    if setupOptions.isXRD:
        spectrumData = SpectrumData(rawData[0], rawData[1], nakedRawFileName)  # Make SpectrumData object and store data in it
    else:  # Convert wavelength to nm
        spectrumData = SpectrumData(wavelengthnmToEnergyEV(rawData[0]), rawData[1], nakedRawFileName)  # Make SpectrumData object and store data in it
    if setupOptions.doBackgroundSubtraction:
        rollingBall = RollingBall()  # Initialize RollingBall object
        backgroundSubtractionPlotting(spectrumData, rollingBall, setupOptions)  # Interactive rolling ball background subtraction
        spectrumData.background = rollingBallBackground(spectrumData, rollingBall.ratio, rollingBall.radius)  # Store the rolling ball background in the SpectrumData object
    else:
        spectrumData.background = list(np.zeros(spectrumData.numXVals))  # Have a zero background, for compatibility with subsequent code
    spectrumData.bgSubIntensity = spectrumData.lnIntensity - spectrumData.background  # Store the background subtracted intensity (natural log) in the SpectrumData object
    spectrumData.expBgSubIntensity = np.exp(spectrumData.bgSubIntensity)  # Store the background subtracted intensity (as measured) in the SpectrumData object
    roiCoordsList, multiRegionCoordsList = fittingRegionSelectionPlotting(spectrumData, setupOptions)  # Interactive region of interest (ROI) selection for fitting
    snContentFittingPlotting(spectrumData, roiCoordsList, multiRegionCoordsList, setupOptions)  # Plot, fit, do PL/CL or XRD specific corrections, and display Sn Contents


if __name__ == "__main__":
    main()
