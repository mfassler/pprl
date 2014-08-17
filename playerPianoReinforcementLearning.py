#!/usr/bin/python
# -*- coding: utf-8 -*-

# Echoic memory, working memory, action selection and reinforcement learning, all
# in one, single algorithm.  

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import os
import sys
import time
import socket
import select
import struct

import numpy as np

from libs.RingBuffer import RingBuffer

# The GUI to display what's in the "mind" of our machine.  Right now,
# only the pygame version works, but it should be possible to build a
# Tkinter version...
from libGUI.MatrixDisplayThing_pygame import MatrixDisplayThing
#from libGUI.MatrixDisplayThing_Tkinter import MatrixDisplayThing
#from libGUI.MatrixDisplayThing_matplotlib import MatrixDisplayThing
mdt = MatrixDisplayThing()


# The Long-Term Memory that learns.  Right now, there's a neural-network
# version and a template-matching version.  Neither of them gives satisfactory
# results....  This is the main WIP:
#from libLTM.LongTermMemory_nn import LongTermMemory
#ltm = LongTermMemory("data_LTM/nnMemories_002.pkl")

from libLTM.LongTermMemory_templates import LongTermMemory
ltm = LongTermMemory("data_LTM/templateMemories_007long.pkl")


import logging
logging.basicConfig(level=logging.DEBUG)




class IoLoop():
    def __init__(self, portNumber):
        self.tLastParse = time.time()
        self.port = portNumber
        self.sd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sd.bind(('127.0.0.1', self.port))
        print "Listening on port#", self.port
    def waitForInput(self):
        inputs, outputs, errors = select.select([self.sd], [], [])
        if inputs or outputs or errors:
            for oneInput in inputs:
                if oneInput == self.sd:
                    onePacket = self.sd.recv(1500)
                    return self.parseInput(onePacket)
        else:
            pass
            #print "timeout... hohum"
    def parseInput(self, onePacket):
        try:
            myCodez = struct.unpack('!ffffffffff', onePacket)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            print "failed to parse packet..."
            print "len: ", len(onePacket)
            print "packet: ", onePacket
        else:
            t0 = time.time()
            thePresent = np.array(myCodez)
            return thePresent


# because I'm always too dumb to remember:
def expDecay(t, tau):
    return np.exp(-t/tau)

def getTau(t, decayPerTimestep):
    return -t / np.log(decayPerTimestep)


class PlayerPianoReinforcementLearning():
    def __init__(self):
        self.rBuf = RingBuffer(75, 10, np.float64)
        tStepSize = 0.1
        discountFactor = 0.98
        tau = getTau(tStepSize, discountFactor)
        self.futureDiscount = expDecay(np.arange(224) * tStepSize, tau)
        self.prevExpA = 0.0
        self.prevExpB = 0.0
    def newPresentData(self, thePresent):
        self.rBuf.append(thePresent)
        echoicMem = self.rBuf.get()
        self.echoicMem = echoicMem


        idx = None
        ucRewardA = False
        ucRewardB = False

        # Actually receiving a primary reward (UCS) is special:
        if np.all(echoicMem[0:5, 0] == np.array([1,1,0,0,0])):
            ucRewardA = True
            idx = 74
        elif np.all(echoicMem[0:5, 1] == np.array([1,1,0,0,0])):
            ucRewardB = True
            idx = 74
        elif np.all(echoicMem[0:5, 0] == np.array([1,1,1,0,0])):
            ucRewardA = True
            idx = 73
        elif np.all(echoicMem[0:5, 1] == np.array([1,1,1,0,0])):
            ucRewardB = True
            idx = 73
        elif np.all(echoicMem[0:5, 0] == np.array([0,1,1,1,0])):
            ucRewardA = True
            idx = 72
        elif np.all(echoicMem[0:5, 1] == np.array([0,1,1,1,0])):
            ucRewardB = True
            idx = 72
        elif np.all(echoicMem[0:5, 0] == np.array([0,0,1,1,1])):
            ucRewardA = True
            idx = 71
        elif np.all(echoicMem[0:5, 1] == np.array([0,0,1,1,1])):
            ucRewardB = True
            idx = 71
        elif np.all(echoicMem[0:5, 0] == np.array([0,0,0,1,1])):
            ucRewardA = True
            idx = 70
        elif np.all(echoicMem[0:5, 1] == np.array([0,0,0,1,1])):
            ucRewardB = True
            idx = 70


        idx, recon, error, idx2, recon2, error2 = ltm.lookForMatches(echoicMem, idx=idx)

        if ucRewardA or ucRewardB:
            print " *** Reward.  idx:", idx

        self.reconIdx = idx
        self.recon = recon
        self.reconError = error

        self.recon2Idx = idx2
        self.recon2 = recon2
        self.recon2Error = error2

        # TODO:  I'm pretty sure this is wrong...
        confidence = np.clip(1.0 / error, 0.01, 1.0)
        # TODO:  I'm pretty sure any of the expressions derived from confidence are not too trustworthy...


        expA = self.getExpectedRewardA(recon, idx)
        expB = self.getExpectedRewardB(recon, idx)

        # The change in expected future reward from previous timestep.  If our prediction is the
        # same, then this should increase by 1/discoutFactor.  If our prediction changes, then
        # we should add to the dopamine signal in some way.  (ie, the dopamine signal is reward
        # prediction error)
        deltaExpA = expA - self.prevExpA
        self.prevExpA = expA
        primaryRewardPredictionErrorA = echoicMem[0, 0] - recon[idx, 0]
        learningRateA = deltaExpA *confidence + primaryRewardPredictionErrorA

        deltaExpB = expB - self.prevExpB
        self.prevExpB = expB
        primaryRewardPredictionErrorB = echoicMem[0, 1] - recon[idx, 1]
        learningRateB = deltaExpB *confidence + primaryRewardPredictionErrorB

        if idx < 73:
            # TODO:  change to:  no learning while there is a reward in echoic memory (t>3 or something)
            self.learningRate = 0.0 # there is no learning after a reward (to preserve the lead-out)
        else:
            self.learningRate = (learningRateA + learningRateB) * 0.1

        # TODO, maybe:  recon and idx already came from ltm, and echoicMem was already provided to
        # ltm, so within the paradigm of classical oo-programming, we could just reuse these...  
        # (probably zero impact on performance, though...)
        ltm.learnMeSomeData(recon, idx, echoicMem, self.learningRate)

    def getExpectedRewardA(self, template, pos):
        return np.dot( (self.futureDiscount[0:pos])[::-1], template[0:pos, 0])

    def getExpectedRewardB(self, template, pos):
        return np.dot( (self.futureDiscount[0:pos])[::-1], template[0:pos, 1])



pprl = PlayerPianoReinforcementLearning()

ioloop = IoLoop(22119)

tLast = time.time()

while True:
    thePresent = ioloop.waitForInput()

    t0 = time.time()
    if (t0 - tLast) > 0.15:
        print "WARNING:  ******* One step is taking too long: %f s" % (t0-tLast)
    tLast = t0

    pprl.newPresentData(thePresent)

    t1 = time.time()

    redGreen = pprl.learningRate * 5
    if redGreen > 1.0:
        redGreen = 1.0
    elif redGreen < -1.0:
        redGreen = -1.0
    mdt.dataFromPlayerPiano(pprl.echoicMem,
                            pprl.recon, pprl.reconIdx, pprl.reconError,
                            pprl.recon2, pprl.recon2Idx, pprl.recon2Error,
                            pprl.learningRate, pprl.prevExpA, pprl.prevExpB)

    t2 = time.time()

    print "\x1b[34m",  # change font color to blue (no carriage return)
    print "t1-t0 (pprl): %0.5f  t2-t1 (display): %0.5f " % (t1-t0, t2-t1),
    print "\x1b[0m"  # reset font color (with carriage return)


