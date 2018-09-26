#!/usr/bin/python
# coding:utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

sum_bow_vector = np.loadtxt(
    '/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Bowvectors/BowVec0.txt')
idx = 1
while (1):
    path = '/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Bowvectors/BowVec' + str(
        idx) + '.txt'
    if (not os.path.exists(path)):
        break
    datamat = np.loadtxt(path)
    datamat = np.ceil(datamat)
    sum_bow_vector += datamat
    idx += 1
    print(idx)
# calculate the word whose Frequency of occurrence more than Threshold
threshold=5
idx_over_threshold=[]
for i in range(0,sum_bow_vector.size):
    if sum_bow_vector[i]>threshold:
        idx_over_threshold.append(i)
        print(i)
np.savetxt("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/idx_over_threshold.txt", idx_over_threshold, fmt='%d')
# plot the sum of bow vector
plt.figure(1)
plt.plot(sum_bow_vector, "r")
plt.xlabel('Index of SGD word')
plt.ylabel('Frequency of occurrence')
plt.show()
