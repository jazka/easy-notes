#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-18 22:52:16
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

def step(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def main():
    x = np.array([-1.0, 1.0, 2.0])
    print(step(x))
    print(sigmoid(x))
    print(relu(x))

if __name__ == '__main__':
    main()
