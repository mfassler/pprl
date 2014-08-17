#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import OneLayerLogistic


class LongTermMemory():
    def __init__(self, pklFileName):
        self.nn = OneLayerLogistic()
        self.filename = pklFileName
        self.nn.load(self.filename)

    def lookForMatches(self, echoicMemory, idx=None):
        assert echoicMemory.shape == (75, 10)

        # The echoicMemory shape is 75 rows of 10 neurons
        # the reconstruction shape is 224 rows of 10 neurons
        maxOffset = 149

        offsets = np.arange(maxOffset)
        errors = np.zeros(maxOffset)
        hidprobs = np.zeros((maxOffset, self.nn.w.shape[1]))

        if idx == None:
            for i in xrange(maxOffset):
                flattenedOffset = i * echoicMemory.shape[1]

                hidprobs[i, :] = self.nn.upPartial(echoicMemory.reshape(1, 750), flattenedOffset)
        else:
            flattenedOffset = idx * echoicMemory.shape[1]

            hidprobs[idx, :] = self.nn.upPartial(echoicMemory.reshape(1, 750), flattenedOffset)

        # TODO:  if an idx is provided, we can save a *lot* of CPU cycles here...
        randomNumbers = np.random.rand(*hidprobs.shape)
        hidstates = 1 * (hidprobs > randomNumbers)
        del randomNumbers

        # This is the slowest part:
        # TODO:  if an idx is provided, we can save a *lot* of CPU cycles here...
        global recons
        recons = self.nn.down(hidstates)

        if idx == None:
            for i in xrange(maxOffset):
                riStart = i*10
                riStop = (i+75)*10
                myErrors = echoicMemory.flatten() - recons[i, riStart:riStop]
                squaredErrors = myErrors**2
                sumOfSquaredErrors = squaredErrors.sum()
                errors[i] = 0.5 * sumOfSquaredErrors
            idx = np.argmin(errors)
        else:
            riStart = idx*10
            riStop = (idx+75)*10
            myErrors = echoicMemory.flatten() - recons[idx, riStart:riStop]
            squaredErrors = myErrors**2
            sumOfSquaredErrors = squaredErrors.sum()
            errors[idx] = 0.5 * sumOfSquaredErrors


        return idx, recons[idx].reshape(224, 10), errors[idx]


    def learnMeSomeData(self, basePattern, idx, echoicMem, learningRate):
        trainingData = np.zeros(basePattern.shape)
        trainingData[idx:idx+75] = echoicMem * 1.1 - 0.1 # Null events will have slightly negative value
        self.nn.cd1(trainingData.flatten(), epsilon=learningRate)

