#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-20 06:43:24
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def cross_entroy_error_batch(y, t, one_hot=False):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot:
        return -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def main():
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    print(mean_squared_error(y, t))
    print(cross_entropy_error(y, t))
    print(cross_entroy_error_batch(y, t))

if __name__ == '__main__':
    main()
