#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-25 18:48:40
# @Author  : Jazka (jazka@testin.cn)
# @Link    : http://www.testin.cn
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet_8 import DeepConvNet
from common.trainer import Trainer


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)
    network = DeepConvNet()
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
    trainer.train()

    network.save_params('deep_convnet_params.pkl')
    print("Saved Network Parameters!")

if __name__ == '__main__':
    main()
