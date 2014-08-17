#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2014, Mark Fassler
# Licensed under the GPLv3


import os
import sys
import time
import socket
import struct

import numpy as np

from libs.PicLoader import loadTrainingImages

sd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sd.connect(('127.0.0.1', 22119))

# clear out any old data on the socket:
sd.setblocking(0)
nothing = 1
while (nothing):
    try:
        nothing = sd.recv(1522)
    except:
        break
sd.setblocking(1)
# ... incoming data should be empty

allPics, allPicNames = loadTrainingImages()

for i in xrange(len(allPicNames)):
    print allPicNames[i]

while True:
    for i in xrange(len(allPicNames)):
        time.sleep(0.3)
        print
        print " *** Starting: ", allPicNames[i]
        print
        time.sleep(0.5)
        for rowNumber in xrange(224, 0, -1):
            outPacket = struct.pack("!ffffffffff", *allPics[i][rowNumber-1])
            sd.send(outPacket)
            time.sleep(0.13)


