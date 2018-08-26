#!/usr/bin/python
# coding:utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
sum_bow_vector = np.loadtxt('/home/ubuntu/Desktop/ch12/BowVec0.txt')
idx=1
while(1):
    path='/home/ubuntu/Desktop/ch12/BowVec' + str(idx) + '.txt'
    if(not os.path.exists(path)):
        break
    datamat = np.loadtxt('/home/ubuntu/Desktop/ch12/BowVec'+str(idx)+'.txt')
    datamat=np.ceil(datamat)
    sum_bow_vector+=datamat
    idx+=1
plt.figure(1)
plt.plot(sum_bow_vector)
plt.xlabel('Index of SGD word')
plt.ylabel('Frequency of occurrence')
plt.show()

