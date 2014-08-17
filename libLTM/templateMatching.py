#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np
import random
import time
import cPickle as pickle

# Brute-force template matching 

class TemplateMemories():
    def __init__(self):
        self.inputShape = (75, 10)
        self.templateShape = (300, 10)
        assert self.inputShape[1] == self.templateShape[1]
        self.maxNumberOfRowsToSearch = 225

    def load(self, filename):
        fd = open(filename, 'rb')
        self.templates = pickle.load(fd)
        fd.close()

    def save(self, filename):
        fd = open(filename, 'wb')
        pickle.dump(self.templates, fd)
        fd.close()

    def loadTemplates(self):
        flatInputSize = self.inputShape[0] * self.inputShape[1]

        self.w = np.zeros((flatInputSize, len(self.templates) * self.maxNumberOfRowsToSearch))

        for i in xrange(len(self.templates)):
            for j in xrange(self.maxNumberOfRowsToSearch):
                col = i * self.maxNumberOfRowsToSearch + j

                self.w[:, col] = self.templates[i][j:j+self.inputShape[0], :].flatten()

    def searchMemory(self, inputSample, rowIdx=None):
        mySumOfSquaredErrors = np.zeros(self.w.shape[1])
        if rowIdx:
            for i in xrange(len(mySumOfSquaredErrors)):
                if (i % self.maxNumberOfRowsToSearch) != rowIdx:
                    mySumOfSquaredErrors[i] = 9999
                    continue
                myErrors = self.w[:, i] - inputSample.flatten()
                squaredErrors = myErrors ** 2
                mySumOfSquaredErrors[i] = squaredErrors.sum()
        else:
            myErrors = self.w - np.tile(inputSample.reshape((750,1)), (1, 4500))
            squaredErrors = myErrors ** 2
            mySumOfSquaredErrors = squaredErrors.sum(0)

        idx = np.argmin(mySumOfSquaredErrors)
        tmp = mySumOfSquaredErrors.copy()
        tmp[idx] = mySumOfSquaredErrors.max() + 0.1
        idx2 = np.argmin(tmp)

        templateIdx = idx / self.maxNumberOfRowsToSearch
        rowIdx = idx % self.maxNumberOfRowsToSearch
        templateIdx2 = idx2 / self.maxNumberOfRowsToSearch
        rowIdx2 = idx2 % self.maxNumberOfRowsToSearch
        return templateIdx, rowIdx, mySumOfSquaredErrors[idx], templateIdx2, rowIdx2, mySumOfSquaredErrors[idx2]


if __name__ == "__main__":
    tMem = TemplateMemories()

    tMem.load('randomPatternsNoRewardsOrActions/20noisePatternsNoRewardsOrActions.pkl')
    tMem.loadTemplates()

    def randomTest():
        i = random.randrange(0, len(tMem.templates))
        pos = random.randrange(0, 169)

        exampleInput = tMem.templates[i][pos:pos+20, :]

        t0 = time.time()
        a, b = tMem.searchMemory(exampleInput)
        t1 = time.time()

        print "t:", t1-t0,
        if a == i and b == pos:
            print " Correct"
        else:
            print " Incorrect"

    for i in xrange(200):
        randomTest()

    # Matching against 20 possible templates is about 0.7 ms  :-)
    # Matching against 350 possible templates is about 10 ms


