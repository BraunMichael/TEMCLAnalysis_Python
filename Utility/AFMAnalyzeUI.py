import os
import json
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Utility.Utilities import *
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class TextValidator(object):
    def __init__(self, tkWindow, minimumScaleBarWidthMicronsValue, maximumScaleBarWidthMicronsValue):
        self.tkWindow = tkWindow
        self.minimumScaleBarWidthMicronsValue = minimumScaleBarWidthMicronsValue
        self.maximumScaleBarWidthMicronsValue = maximumScaleBarWidthMicronsValue

    def stringNumberRangeValidator(self, proposedText, minimumValue, maximumValue):
        if proposedText == '':
            return True
        if not proposedText.replace('.', '', 1).isdigit():
            self.tkWindow.bell()
            return False
        numberFloat = strToFloat(proposedText)
        if minimumValue <= numberFloat <= maximumValue:
            return True
        self.tkWindow.bell()
        return False

    def scaleBarWidthMicronsValidator(self, proposedText):
        return self.stringNumberRangeValidator(proposedText, self.minimumScaleBarWidthMicronsValue, self.maximumScaleBarWidthMicronsValue)


def get_file(entryField, entryFieldText, titleMessage, fileFormatsStr):
    listName = getFileOrDir('file', titleMessage, fileFormatsStr, entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def preview_image(imageFilePath):
    inputImagePath = imageFilePath.get().replace('~', os.path.expanduser('~'))
    if inputImagePath.endswith(('.tiff', '.tif')):
        print("attempting to convert tiff to png")
        rawImage = Image.open(inputImagePath)
        npImage = ((np.array(rawImage) + 1) / 256) - 1
        visImage = Image.fromarray(np.uint8(npImage), mode='L')
        visImage.show()
    else:
        rawImage = Image.open(inputImagePath)
        rawImage.show()


def get_setupOptions(savedJSONFileName):
    try:
        with open(savedJSONFileName) as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleBarWidthMicronsVar):
    setupOptions.dataFilePath = ImageEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.scaleBarWidthMicrons = strToFloat(scaleBarWidthMicronsVar.get())

    with open(savedJSONFileName, 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions, savedJSONFileName):
    win.title("AFM Analysis UI")
    ImageEntryText = tkinter.StringVar(value=setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~'))

    scaleBarWidthMicronsVar = tkinter.StringVar(value=setupOptions.scaleBarWidthMicrons)

    tkinter.Label(win, text="Image File:").grid(row=0, column=0)
    ImageFileEntry = tkinter.Entry(win, textvariable=ImageEntryText, width=len(setupOptions.dataFilePath.replace(os.path.expanduser('~'), '~')))
    ImageFileEntry.grid(row=1, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(ImageFileEntry, ImageEntryText, 'Choose Image File', '.tiff .tif')).grid(row=1, column=1)
    tkinter.Button(win, text='Preview', command=lambda: preview_image(ImageEntryText)).grid(row=1, column=2)

    txtValidator = TextValidator(win, minimumScaleBarWidthMicronsValue=0, maximumScaleBarWidthMicronsValue=1000)
    scaleBarWidthMicronsValidatorFunction = (win.register(txtValidator.scaleBarWidthMicronsValidator), '%P')
    tkinter.Label(win, text="Scale Bar Size (Microns)").grid(row=2, column=0)
    tkinter.Entry(win, textvariable=scaleBarWidthMicronsVar, validate='all', validatecommand=scaleBarWidthMicronsValidatorFunction).grid(row=2, column=1)

    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleBarWidthMicronsVar))
    win.mainloop()


def setupOptionsUI():
    savedJSONFileName = 'AFMAnalysisSetupOptions.json'
    setupOptions = get_setupOptions(savedJSONFileName)  # Read previously used setupOptions
    uiInput(Tk(), setupOptions, savedJSONFileName)
    return setupOptions


# setupOptionsUI()