#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-18 23:35:03
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import sys, os
sys.path.append(os.pardir)

import numpy as np
import pickle

from activation_function_3 import *
from output_function_3 import *
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,
            one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('weight/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def main():
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    # for i in range(len(x)):
    #     y = predict(network, x[i])
    #     p = np.argmax(y)
    #     if p == t[i]:
    #         accuracy_cnt += 1

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i: i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

if __name__ == '__main__':
    main()
