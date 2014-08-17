#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image


def loadTrainingImages():
    allTrainingImages = glob.glob('training_10x224/*.png')

    allPics = []
    allPicNames = []

    for oneFile in allTrainingImages:
        img = Image.open(oneFile).convert("L")
        imgDat = np.array(img, np.float)
        # these have white (255) background, so let's invert
        imgDatInverse = 255 - imgDat
        imgDatInverse /= 255.0
        allPics.append(imgDatInverse)
        allPicNames.append(os.path.basename(oneFile))

    return allPics, allPicNames





