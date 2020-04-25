# XRDProcessing
XRD Data processing including background subtraction and peak fitting

For filetypes, it can do tab separated, comma separated, space separated 2+ column txt, xy, dat, csv files. Strings in the header are ok, but eventually the numeric data needs to be 2theta and then intensity (normalized intensity is ok).

For each plot that pops up, once you are happy with the outcome (ie good background subtracted, or selected all the regions you want) just close the plot window and the script will take the values and continue. For region selecting, click and drag (will be red), then when you hit add region it will save that region of interest (turn it green) and allow you to drag and select another region.
