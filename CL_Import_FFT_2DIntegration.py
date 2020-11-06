import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, Slider
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString, MultiLineString, Polygon

pixelScale = 49  #nm per pixel
averagedSlices = 5

def radial_profile(data, center):
    # https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

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



rawCL = np.loadtxt('5min_Sample2.txt')
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

fftManager = FFTManager(fig, ax)
axFFT = plt.axes([0.7, 0.02, 0.2, 0.075])
bFFT = Button(axFFT, 'Calc FFT')
rect = RectangleSelector(ax, fftManager.RangeSelection, drawtype='box', rectprops=dict(facecolor='none', edgecolor='red', alpha=0.5, fill=False))
bFFT.on_clicked(fftManager.FFTButtonClicked)

plt.show()
centerSliceIndex = int(sSlice.val)
centerWavelengthValue = wavelengths[centerSliceIndex]
coordsOut = fftManager.coords

# TODO: Pull avereagedOut data from coordsOut, FFT then radial integrate
plt.close()

xMin = int(round(coordsOut['x'][0]))
xMax = int(round(coordsOut['x'][1]))
yMin = int(round(coordsOut['y'][0]))
yMax = int(round(coordsOut['y'][1]))
assert xMax-xMin == yMax-yMin, "The selected FFT area does not appear to be square, make sure to hold shift when selecting the area of interest"
croppedCL = outAveraged[centerSliceIndex, yMin:yMax, xMin:xMax]

_, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
# TODO: use pcolormesh instead to set scaled axes https://stackoverflow.com/questions/34003120/matplotlib-personalize-imshow-axis
plt.imshow(croppedCL, interpolation='none', cmap='plasma', norm=LogNorm())
ax.margins(x=0)
plt.axis('equal')
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


print('here')