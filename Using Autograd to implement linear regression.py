#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-09-01 15:30:51
# @Author  : mutudeh (josephmathone@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import os
import torch as t
import matplotlib.pyplot as plt
import numpy as np


t.manual_seed(1000) 
# the real function is y = 3*x + 2
def retrive_fake_data(batch_size = 8):
	x = t.rand(batch_size,1)*5
	y = 3 * x + 2 + t.rand(batch_size,1)*2
	return x,y

x,y = retrive_fake_data(10)
plt.scatter(x,y,marker = 'o')


w = t.rand(1,1,requires_grad = True)
b = t.zeros(1,1,requires_grad = True)
lr = 0.005

loss = 0
total_loss = []

batch_size = 32
for i in range(50):
	loss = 0
	x,y = retrive_fake_data(batch_size)

	y_pre = x.mm(w) + b.expand_as(y)
	loss = 0.5 * (y-y_pre) ** 2
	loss = loss.sum() / batch_size

	loss.backward()

	w.data.sub_(lr*w.grad.data)
	b.data.sub_(lr*b.grad.data)

	w.grad.data.zero_()
	b.grad.data.zero_()


	total_loss.append(loss)


print(total_loss)
# plt.plot(total_loss)
# plt.ylim(5,50)
plt.show()
