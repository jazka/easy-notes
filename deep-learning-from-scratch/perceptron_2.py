#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-18 22:34:23
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    return 0 if tmp <= 0 else 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

def main():
    # print(AND(0, 0))
    # print(AND(0, 1))
    # print(AND(1, 0))
    # print(AND(1, 1))
    # print(OR(0, 0))
    # print(OR(0, 1))
    # print(OR(1, 0))
    # print(OR(1, 1))
    # print(NAND(0, 0))
    # print(NAND(0, 1))
    # print(NAND(1, 0))
    # print(NAND(1, 1))
    print(XOR(0, 0))
    print(XOR(0, 1))
    print(XOR(1, 0))
    print(XOR(1, 1))

if __name__ == '__main__':
    main()
