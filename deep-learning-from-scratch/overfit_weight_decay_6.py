#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-23 01:12:43
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    x_train = x_train[:300]
    t_train = t_train[:300]

    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01
    # weight_decay_lambda = 0
    weight_decay_lambda = 0.1

    train_acc_list = []
    test_acc_list = []

    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                                    output_size=10, weight_decay_lambda=weight_decay_lambda)
    optimizer = SGD(lr=learning_rate)

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + \
                    ", test acc:" + str(test_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
