#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-22 11:17:51
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np

class MulLayer:
    """docstring for MulLayer"""
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    """docstring for AddLayer"""
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy


def main():
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print(price)

    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple, dapple_num, dtax)

if __name__ == '__main__':
    main()
