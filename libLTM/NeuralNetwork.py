#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np
import cPickle as pickle
import time

import logging


def logsig(t):
    return 1.0 / (1.0 + np.exp(-t))


def logsigLinearMiddle(t):
    return 4.0 / (1.0 + np.exp(-t)) - 2.0

class OneLayerLogistic:
    def __init__(self):
        self.w = np.zeros((2240, 500))
        self.vB = np.zeros(2240)
        self.hB = np.zeros(500)

    def initW(self):
        self.w = 0.005 * np.random.randn(2240, 500)

    def save(self, filename):
        fd = open(filename, 'wb')
        pickle.dump({'w': self.w, 'vB': self.vB, 'hB': self.hB}, fd, protocol=-1)
        fd.close()

    def load(self, filename):
        fd = open(filename, 'rb')
        tmp = pickle.load(fd)
        fd.close()
        self.w = tmp['w']
        self.vB = tmp['vB']
        self.hB = tmp['hB']

    def up(self, inputData):
        return logsig(np.dot(inputData, self.w) + self.hB)

    # The echoic memory is 75x10.  It should be reshaped to 1x750
    def upPartial(self, inputData, offset):
        assert len(inputData.shape) == 2, "must be a 2-D matrix"

        # Find the common overlap of two matrices:
        idLen = inputData.shape[1]

        wStart = np.clip(offset, 0, self.w.shape[0])
        wStop = offset + idLen
        wStop = np.clip(wStop, 0, self.w.shape[0])
        overlap = wStop - wStart

        idStart = wStart - offset
        idStop = idStart + overlap

        #  nn.up of just the partial overlap:
        return logsig(np.dot(inputData[:, idStart:idStop], self.w[wStart:wStop :]) + self.hB)

    def sup(self, inputData, randomNumbers=None):
        # stochastic up
        hidprobs = self.up(inputData)
        if (randomNumbers == None):
            randomNumbers = np.random.rand(*hidprobs.shape)
        hidstates = 1 * (hidprobs > randomNumbers)
        del randomNumbers
        return hidstates

    def down(self, inputData):
        return logsig(np.dot(inputData, self.w.T) + self.vB)

    def sdown(self, inputData, randomNumbers=None):
        # stochastic down
        visprobs = self.down(inputData)
        if (randomNumbers == None):
            randomNumbers = np.random.rand(*visprobs.shape)
        visstates = 1 * (visprobs > randomNumbers)
        del randomNumbers
        return visstates

    def cd1(self, inputData, epsilon=0.1, randomNumbers=None):
        if len(inputData.shape) == 1:
            inputData = np.array([inputData])

        # Hinton's contrastive divergence, 1-step.  (Or my poor understanding therof, anyway...)
        # Our only goal is to reconstruct known patterns based on incomplete data.  
        epsilonW = epsilon
        epsilonVB = epsilon
        epsilonHB = epsilon
        weightcost = 0.2

        # The "up" (or "positive") phase:
        poshidprobs = self.up(inputData)

        if (randomNumbers == None):
            randomNumbers = np.random.rand(*poshidprobs.shape)
        poshidstates = 1 * (poshidprobs > randomNumbers)
        del randomNumbers
        positiveProducts = np.dot(inputData.T, poshidprobs)  #an unbiased sample, <v_i, h_j>_data

        # The "down-up" (or "negative") phase:
        logging.debug("phs.shape: %s   epsilon: %d" % (poshidstates.shape, epsilon))
        negdata = self.down(poshidstates) # reconstruction
        neghidprobs = self.up(negdata)
        negativeProducts = np.dot(negdata.T, neghidprobs)

        if len(inputData.shape) == 1:
            numcases = 1
        else:
            numcases = inputData.shape[0]
        posvisact = inputData.sum(0)
        poshidact = poshidprobs.sum(0)
        negvisact = negdata.sum(0)
        neghidact = neghidprobs.sum(0)

        #deltaW = epsilonW * ( (positiveProducts-negativeProducts)/numcases - weightcost * self.w)
        deltaW = epsilonW * ( (positiveProducts-negativeProducts)/numcases )

        deltaVB = (epsilonVB / numcases) * (posvisact - negvisact)
        deltaHB = (epsilonHB / numcases) * (poshidact - neghidact)

        self.w = logsigLinearMiddle(self.w + deltaW)
        self.vB = logsigLinearMiddle(self.vB + deltaVB)
        self.hB = logsigLinearMiddle(self.hB + deltaHB)
        #self.vB += deltaVB
        #self.hB += deltaHB

        # Order of magnitude:
        #  in the Eruditio/rbm-cd example, the ranges are:
        #   -2.0 < W < 2.0
        #   -0.6 < vB < 0.6
        #   -5.0 < hB < 5.0
        # all the training data and targets are scaled 0.0 to 1.0 (not 0 -- 255)

