#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np
import pygame

from libs.RingBuffer import RingBuffer


class MatrixDisplayThing():
    def __init__(self):
        # echoic memory is 75 rows.  PlayerPianoRoll memory is 300 rows, 75 of which
        # must overlap with echoic memory.
        # So:  overlap is 75 rows.  Max future overshoot is 300-75.  Max past 
        # overshoot is 300-75 rows.  So total display rows is: 525
        # NUMPY matrices are row, column,
        # but screen resolutions are width, height  (ie, transposed)
        self.width = 1050
        self.height = 250
        self.screen = pygame.display.set_mode((self.width, self.height), 0)
        pygame.display.set_caption("Player Piano Reinfocement Learning")
        self.offsetA = 0
        self.offsetB = 0
        self.shapeA = (0,0)
        self.shapeB = (0,0)
        self.unknownColor = (150,200,230)
        self.screen.fill(self.unknownColor)
        self.unknownColorLine = np.dstack(( np.ones((10)) * 150, np.ones((10)) * 200, np.ones((10)) * 230 ))

        self.fontColor_reddish = (150, 63,0)
        pygame.font.init()
        self.myfont = pygame.font.SysFont("arial", 15)
        self.mySmallFont = pygame.font.SysFont("arial", 9)

        # Graph and text for learning rate
        self.textSurface_learningRate = self.myfont.render("0.000", 1, self.fontColor_reddish)

        # Text for expected future rewards
        self.textSurface_futureRewards = self.mySmallFont.render("Expected Future Rewards: A: , B: ", 1, self.fontColor_reddish)

        self.learningRateHistory = RingBuffer(150, 40, np.uint8)
        self.lrhWhite = np.ones((150,40), np.uint8) * 255
        for i in xrange(150):
            self.learningRateHistory.append(np.ones(40) * 255)
        self.learningRateSurface = pygame.Surface((150, 40))

        # The echoic memory pattern:
        self.echoicSurface = pygame.Surface((75, 10))
        self.echoicSurfaceScaled = pygame.Surface((75*2, 10*2))

        # The reconstructed patterns:
        self.textSurface_reconError = self.myfont.render("Best reconstruction", 1, self.fontColor_reddish)
        self.textSurface_recon2Error = self.myfont.render("Second-best reconstruction", 1, self.fontColor_reddish)
        self.reconBuffer = np.zeros((525, 10, 3), np.uint8)
        self.recon2Buffer = np.zeros((525, 10, 3), np.uint8)
        self.reconSurface = pygame.Surface((525, 10))
        self.recon2Surface = pygame.Surface((525, 10))
        self.reconSurfaceScaled = pygame.Surface((525*2, 10*2))
        self.recon2SurfaceScaled = pygame.Surface((525*2, 10*2))
        self.prevXStart1 = 0
        self.prevXStop1 = 524
        self.prevXStart2 = 0
        self.prevXStop2 = 524

        self.labels = {}
        self.labels['learningRate'] = self.myfont.render(u"← Learning rate", 1, self.fontColor_reddish)
        self.labels['echoicMem'] = self.myfont.render(u"← Echoic memory", 1, self.fontColor_reddish)

        self.screen.blit(self.labels['learningRate'], (720, 70))
        self.screen.blit(self.labels['echoicMem'], (720, 110))

        self.drawTimeScale()
        self.updateDisplay()

    def drawTimeScale(self):
        zeroXpoint = 600
        tTop = 35
        htTop = 42
        tBottom = 50
        pygame.draw.line(self.screen, (0,0,0), (zeroXpoint, 0), (zeroXpoint, self.height))
        pygame.draw.line(self.screen, (0,0,0), (0, tBottom), (self.width, tBottom))
        # Every second into the future:
        for i in xrange(zeroXpoint, self.width, 20):
            pygame.draw.line(self.screen, (0,0,0), (i, tTop), (i, tBottom))
        # Every half-second into the future:
        for i in xrange(zeroXpoint+10, self.width, 20):
            pygame.draw.line(self.screen, (0,0,0), (i, htTop), (i, tBottom))
        # Every second into the past:
        for i in xrange(zeroXpoint, 0, -20):
            pygame.draw.line(self.screen, (0,0,0), (i, tTop), (i, tBottom))
        # Every half-second into the past:
        for i in xrange(zeroXpoint-10, 0, -20):
            pygame.draw.line(self.screen, (0,0,0), (i, htTop), (i, tBottom))

        self.labels['thePast'] = self.myfont.render(u"The Past (seconds) ←", 1, self.fontColor_reddish)
        self.labels['theFuture'] = self.myfont.render(u"→ The Future (seconds)", 1, self.fontColor_reddish)
        self.screen.blit(self.labels['thePast'], (zeroXpoint - 150, 0))
        self.screen.blit(self.labels['theFuture'], (zeroXpoint + 5, 0))

        self.labels['5'] = self.mySmallFont.render(u"5", 1, (0,0,0))
        self.labels['10'] = self.mySmallFont.render(u"10", 1, (0,0,0))
        self.screen.blit(self.labels['5'], (zeroXpoint - 103, tTop - 10))
        self.screen.blit(self.labels['10'], (zeroXpoint - 204, tTop - 10))
        self.screen.blit(self.labels['5'], (zeroXpoint + 97, tTop - 10))
        self.screen.blit(self.labels['10'], (zeroXpoint + 196, tTop - 10))
        

    def updateDisplay(self):
        
        self.screen.fill(self.unknownColor, (620, 70, 100, 40))
        self.screen.blit(self.textSurface_learningRate, (620, 70))
        self.screen.blit(self.learningRateSurface, (450, 60))
        self.screen.blit(self.echoicSurfaceScaled, (450, 110))

        self.screen.fill(self.unknownColor, (610, 140, 400, 40))
        self.screen.blit(self.textSurface_futureRewards, (610, 140))

        self.screen.fill(self.unknownColor, (20, 140, 400, 40))
        self.screen.blit(self.textSurface_reconError, (20, 140))
        self.screen.blit(self.reconSurfaceScaled, (2, 160))

        self.screen.fill(self.unknownColor, (20, 190, 400, 40))
        self.screen.blit(self.textSurface_recon2Error, (20, 190))
        self.screen.blit(self.recon2SurfaceScaled, (2, 210))
        pygame.display.flip()

    def dataFromPlayerPiano(self, echoicMem, recon1, recon1Offset, recon1Error, recon2, recon2Offset, recon2Error, learningRate, expA, expB):
        self.updateLearningRate(learningRate)
        self.updateExpFutureRewards(expA, expB)
        self.updateEchoicMem(echoicMem)
        self.updateRecon1Mem(recon1, recon1Offset, recon1Error, learningRate)
        self.updateRecon2Mem(recon2, recon2Offset, recon2Error)
        self.updateDisplay()

    def updateLearningRate(self, learningRate):
        nextRow = np.ones(40) * 255
        blackDotPos = int(-learningRate * 105 + 20)
        if blackDotPos > 39:
            blackDotPos = 39
        elif blackDotPos < 0:
            blackDotPos = 0
        nextRow[blackDotPos] = 0 

        # append two rows because we're operating at twice the resolution of the other surfaces:
        self.learningRateHistory.append(nextRow)
        self.learningRateHistory.append(nextRow)

        lrh = self.learningRateHistory.get()
        flipAndStack = np.dstack((lrh[::-1], lrh[::-1], self.lrhWhite))
        pygame.surfarray.blit_array(self.learningRateSurface, flipAndStack)

        # also show the literal text of the current value:
        self.textSurface_learningRate = self.myfont.render("%f" % (learningRate), 1, self.fontColor_reddish)

    def updateExpFutureRewards(self, expA, expB):
        self.textSurface_futureRewards = self.mySmallFont.render("Expected Future Rewards: A: %f,   B: %f" % (expA, expB), 1, self.fontColor_reddish)

    def updateEchoicMem(self, echoicMem):
        assert echoicMem.shape == (75, 10)
        intMatrix = ((1.0 - echoicMem) * 255).astype(np.uint8)[::-1]
        pygame.surfarray.blit_array(self.echoicSurface, np.dstack((intMatrix, intMatrix, intMatrix)))
        pygame.transform.scale(self.echoicSurface, self.echoicSurfaceScaled.get_size(), self.echoicSurfaceScaled)

    def updateRecon1Mem(self, recon, offset, error, learningRate):
        # lazy, for now -- assume a fixed shape:
        assert recon.shape == (300, 10)

        redGreen = learningRate * 7.5
        if redGreen > 1.0:
            redGreen = 1.0
        elif redGreen < -1.0:
            redGreen = -1.0

        # if redGreen is +1, then things should look rather green,
        # if redGreen is -1, then thhings should look rather red
        if redGreen == 0:
            redNess = 255
            grnNess = 255
            bluNess = 255
        elif redGreen < 0:
            redNess = 255
            grnNess = 255 * (1 + redGreen)
            bluNess = 255 * (1 + redGreen)
        else:
            redNess = 255 * (1 - redGreen)
            grnNess = 255
            bluNess = 255 * (1 - redGreen)

        intMatrixB_r = ((1.0 - np.clip(recon, 0.0, 1.0)) * redNess).astype(np.uint8)[::-1]
        intMatrixB_g = ((1.0 - np.abs(recon)) * grnNess).astype(np.uint8)[::-1]
        intMatrixB_b = ((1.0 - np.abs(recon)) * bluNess).astype(np.uint8)[::-1]

        xStart = offset
        xStop = offset + 300

        # erase the previous chunks:
        if self.prevXStart1 < xStart:
            for i in xrange(self.prevXStart1, xStart):
                self.reconBuffer[i] = self.unknownColorLine
        elif self.prevXStop1 > xStop:
            for i in xrange(xStop, self.prevXStop1):
                self.reconBuffer[i] = self.unknownColorLine

        self.prevXStart1 = xStart
        self.prevXStop1 = xStop
 
        self.reconBuffer[xStart:xStop] = np.dstack((intMatrixB_r, intMatrixB_g, intMatrixB_b))
        pygame.surfarray.blit_array(self.reconSurface, self.reconBuffer)
        pygame.transform.scale(self.reconSurface, self.reconSurfaceScaled.get_size(), self.reconSurfaceScaled)

        self.textSurface_reconError = self.myfont.render("Best reconstruction (error: %f):" % error, 1, self.fontColor_reddish)



    def updateRecon2Mem(self, recon, offset, error):
        # lazy, for now -- assume a fixed shape:
        assert recon.shape == (300, 10)

        intMatrixB_r = ((1.0 - np.clip(recon, 0.0, 1.0)) * 255).astype(np.uint8)[::-1]
        intMatrixB_gb = ((1.0 - np.abs(recon)) * 255).astype(np.uint8)[::-1]

        xStart = offset
        xStop = offset + 300

        # erase the previous chunks:
        if self.prevXStart2 < xStart:
            for i in xrange(self.prevXStart2, xStart):
                self.recon2Buffer[i] = self.unknownColorLine
        elif self.prevXStop2 > xStop:
            for i in xrange(xStop, self.prevXStop2):
                self.recon2Buffer[i] = self.unknownColorLine

        self.prevXStart2 = xStart
        self.prevXStop2 = xStop
 
        self.recon2Buffer[xStart:xStop] = np.dstack((intMatrixB_r, intMatrixB_gb, intMatrixB_gb))
        pygame.surfarray.blit_array(self.recon2Surface, self.recon2Buffer)
        pygame.transform.scale(self.recon2Surface, self.recon2SurfaceScaled.get_size(), self.recon2SurfaceScaled)

        self.textSurface_recon2Error = self.myfont.render("Second-best reconstruction (error: %f):" % error, 1, self.fontColor_reddish)



