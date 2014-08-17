#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import numpy as np


class RingBuffer():
    def __init__(self, rows, cols, dtype):
        self.rows = rows
        self.cols = cols
        self.dtype = dtype
        self.rBuffer = np.zeros((self.rows, self.cols), self.dtype)
        self.pos = 0
    def updateView(self):
        tmp = np.vstack(( self.rBuffer[self.pos:self.rows, :], self.rBuffer[0:self.pos, :] ))
        self.view = tmp[::-1, :]
    def append(self, oneRow):
        self.rBuffer[self.pos, :] = oneRow
        self.pos += 1
        if self.pos >= self.rows:
            self.pos = 0
        self.updateView()
    def get(self):
        return self.view


