#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import glob
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image


np.set_printoptions(threshold='nan')


allPicFiles = glob.glob('*.png')

allPics = []

for onePicFile in allPicFiles:
    img = Image.open(onePicFile).convert("L")
    imgDat = np.array(img)
    # these all have white (255) background... se we need to invert
    imgDatInverse = 255 - imgDat
    allPics.append(imgDatInverse)


