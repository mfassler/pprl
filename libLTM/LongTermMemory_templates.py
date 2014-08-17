#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np
from templateMatching import TemplateMemories


class LongTermMemory():
    def __init__(self, pklFileName):
        self.tmem = TemplateMemories()
        self.filename = pklFileName
        self.tmem.load(self.filename)
        self.tmem.loadTemplates()
        self.whichTemplate = 0
        self.pos = 0

    def lookForMatches(self, echoicMem, idx=None):
        idx, pos, error, idx2, pos2, error2 = self.tmem.searchMemory(echoicMem, idx)

        self.whichTemplate = idx
        self.pos = pos
        return pos, self.tmem.templates[idx], error, pos2, self.tmem.templates[idx2], error2

    def learnMeSomeData(self, recon, idx, echoicMem, learningRate):
        idx = self.whichTemplate
        pos = self.pos

        numTemplates = len(self.tmem.templates)
        # The chosen template will learn at the requested rate
        aa = 1-np.abs(learningRate)
        bb = learningRate

        self.tmem.templates[idx][pos:pos+75] = aa * self.tmem.templates[idx][pos:pos+75] + bb * echoicMem

        # All the other templates will un-learn at the requested rate, divided by the total
        # number of templates
        unlearningRate = learningRate / numTemplates
        aa = 1-np.abs(unlearningRate)
        bb = -unlearningRate

        for i in xrange(numTemplates):
            if i == idx:
                continue
            self.tmem.templates[i][pos:pos+75] = aa * self.tmem.templates[i][pos:pos+75] + bb * echoicMem
            

        self.tmem.loadTemplates()

        # TODO:  Unlearning should be stronger with the 2nd-closest match (regardless of rowIdx), etc...

        # TODO:  the lead-out doesn't need learning.  (perhaps this is a bug, actually...)

