#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-24 17:16:56
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt

from common.layers import *
from common.trainer import Trainer
from dataset.mnist import load_mnist
from simple_convnet_7 import SimpleConvNet

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)
    max_epoch = 20

    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
    trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epoch,
                    mini_batch_size=100, optimizer='Adam', optimizer_param={'lr':0.01},
                    evaluate_sample_num_per_epoch=1000)

    trainer.train()
    network.save_params('params.pkl')
    print("Saved Network Parameters!")

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
