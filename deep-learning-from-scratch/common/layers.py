#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-22 11:32:27
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import sys, os
sys.path.append(os.pardir)

import numpy as np

from common.functions import *
from common.util import im2col, col2im

class Relu:
    """docstring for Relu"""
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    """docstring for Sigmoid"""
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx

class Affine:
    """docstring for Affine"""
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)

        return dx

class SoftmaxWithLoss:
    """docstring for SoftmaxWithLoss"""
    def __init__(self):
        self.loss = None
        self.y = None
        # Support one-hot vector
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
