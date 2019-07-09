#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-20 07:19:59
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        gradient[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def _function_2(x):
    return np.sum(x**2)

def main():
    print(numerical_gradient(_function_2, np.array([3.0, 4.0])))

    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(_function_2, init_x=init_x, lr=0.1, step_num=100))

if __name__ == '__main__':
    main()
