#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-18 23:16:20
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

def identity_function(a):
    return a

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def main():
    x = np.array([1.0, 0.5])
    print(identity_function(x))
    a = np.array([0.3, 2.9, 4.0])
    print(softmax(a))

if __name__ == '__main__':
    main()
