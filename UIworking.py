import os
import re
import math
import warnings
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, SpanSelector, Button
from matplotlib.ticker import AutoMinorLocator
from lmfit.models import PseudoVoigtModel, VoigtModel
from bisect import bisect_left

# https://stackoverflow.com/questions/53701552/how-to-get-multiple-radiobutton-values
win = Tk()

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
    with open(xrdFileNameFunc, 'r') as file:
        for line in file:
            splitLine = re.split(regexPattern, line)
            if all([is_number(splitLine[0]), is_number(splitLine[1])]):
                xValsList.append(float(splitLine[0]))
                intensityList.append(float(splitLine[1]))
    return np.asarray([xValsList, intensityList])


def getData():
    root = Tk()
    root.withdraw()
    fileName = filedialog.askopenfilename(title='Choose XRD xy file', filetypes=[('Spectrum xyfile', '.txt .xy .dat .csv')])
    root.destroy()
    if not fileName:
        quit()
    nakedRawFileName = getNakedNameFromFilePath(fileName)
    print("Working on:", nakedRawFileName)
    rawData = readDataFile(fileName)
    return rawData, nakedRawFileName


def submit_values():
    a = " ".join([str(i.get()) for i in values])
    tk.Label(win, text=a).grid()



def getFileOrDirList(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = None, initialDirOrFile: str = os.getcwd()):
    if os.path.isfile(initialDirOrFile) or os.path.isdir(initialDirOrFile):
        initialDir = os.path.split(initialDirOrFile)[0]
    else:
        initialDir = initialDirOrFile
    root = Tk()
    root.withdraw()
    assert fileOrFolder.lower() == 'file' or fileOrFolder.lower() == 'folder', "Only file or folder is an allowed string choice for fileOrFolder"
    if fileOrFolder.lower() == 'file':
        fileOrFolderList = filedialog.askopenfilename(initialdir=initialDir, title=titleStr, filetypes=[(fileTypes + "file", fileTypes)])
    else:  # Must be folder from assert statement
        fileOrFolderList = filedialog.askdirectory(initialdir=initialDir, title=titleStr)
    if not fileOrFolderList:
        fileOrFolderList = initialDirOrFile
    root.destroy()
    return fileOrFolderList


def get_file(entryField, entryFieldText, titleMessage):
    listName = getFileOrDirList('file', titleMessage, '.txt .xy .csv .dat', entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def hide_GeSnPL(win):
    if 'geSnPL_Label' in win.children:
        win.children['geSnPL_Label'].destroy()
        win.children['geSnPL_YesButton'].destroy()
        win.children['geSnPL_NoButton'].destroy()


def show_GeSnPL(win):
    if 'geSnPL_Label' not in win.children:
        item_Label = tk.Label(win, text="[PL Only] GeSn specific calculations?", name='geSnPL_Label')
        item_Label.grid(row=6, column=0)
        r1isGeSnPL = tk.Radiobutton(win, text="Yes", variable=isGeSnPL, value=1, name='geSnPL_YesButton')
        r2isGeSnPL = tk.Radiobutton(win, text="No", variable=isGeSnPL, value=0, name='geSnPL_NoButton')
        r1isGeSnPL.grid(row=6, column=1)
        r2isGeSnPL.grid(row=6, column=2)


def on_closing(win, dataFileEntryText, darkFileEntryText, isXRD, doBackgroundSubtraction, isGeSnPL):
    print(dataFileEntryText.get())
    print(darkFileEntryText.get())
    print(isXRD.get())
    print(doBackgroundSubtraction.get())
    print(isGeSnPL.get())
    win.destroy()


dataFileEntryText = tk.StringVar()
darkFileEntryText = tk.StringVar()

isXRD = tk.BooleanVar(value=True)
doBackgroundSubtraction = tk.BooleanVar(value=True)
isGeSnPL = tk.BooleanVar(value=False)

tk.Label(win, text="Data File:").grid(row=0, column=0)
dataFileEntry = tk.Entry(win, textvariable=dataFileEntryText)
dataFileEntry.grid(row=1, column=0)
tk.Button(win, text='Choose File', command=lambda: get_file(dataFileEntry, dataFileEntryText, 'Choose Data File')).grid(row=1, column=1)

tk.Label(win, text="Dark File:").grid(row=2, column=0)
darkFileEntry = tk.Entry(win, textvariable=darkFileEntryText)
darkFileEntry.grid(row=3, column=0)
tk.Button(win, text='Choose File', command=lambda: get_file(darkFileEntry, darkFileEntryText, 'Choose Dark Scan File')).grid(row=3, column=1)

item_Label = tk.Label(win, text="XRD or PL/CL")
item_Label.grid(row=4, column=0)
r1isXRD = tk.Radiobutton(win, text="XRD", variable=isXRD, value=1, command=lambda: hide_GeSnPL(win))
r2isXRD = tk.Radiobutton(win, text="PL/CL", variable=isXRD, value=0, command=lambda: show_GeSnPL(win))
r1isXRD.grid(row=4, column=1)
r2isXRD.grid(row=4, column=2)

item_Label = tk.Label(win, text="Background subtraction (interactive)?")
item_Label.grid(row=5, column=0)
r1doBgSub = tk.Radiobutton(win, text="Yes", variable=doBackgroundSubtraction, value=1)
r2doBgSub = tk.Radiobutton(win, text="No", variable=doBackgroundSubtraction, value=0)
r1doBgSub.grid(row=5, column=1)
r2doBgSub.grid(row=5, column=2)

win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, dataFileEntryText, darkFileEntryText, isXRD, doBackgroundSubtraction, isGeSnPL))
win.mainloop()
