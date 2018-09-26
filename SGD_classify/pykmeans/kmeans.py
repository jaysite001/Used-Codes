#!/usr/bin/python
# coding:utf-8

import numpy as np
import os
import time
from sklearn.cluster import KMeans

datamat = np.loadtxt(
    '/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SavedDesBA.txt')
socrefile = open('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/inertia.txt', 'w')
num_kinds = 6
start_time = time.time()
estimator = KMeans(n_clusters=num_kinds)  # 构造聚类器
estimator.fit(datamat)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
centroids = estimator.cluster_centers_  # 获取聚类中心
inertia = estimator.inertia_  # 获取聚类准则的最后值
cost_time = time.time() - start_time
socrefile.writelines(str(num_kinds) + ' ' + str(inertia) + ' ' + str(cost_time) + '\n')
np.savetxt("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/orignLabel.txt", label_pred, fmt='%d')
np.savetxt("/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Centers.txt", centroids, fmt='%.6f')
socrefile.close()

# create label for SGD
des_file = open('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SavedDesBA.txt')
original_label_file = open('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/orignLabel.txt')
fixed_label_file = open('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Label.txt',
                        'w')

for line in des_file:
    if len(line) > 1:
        label_line = original_label_file.readline()
    else:
        label_line = str(0) + '\n'
    fixed_label_file.writelines(label_line)

des_file.close()
original_label_file.close()
fixed_label_file.close()

# create Labeled PCD file
original_pcd_file = open('/home/ubuntu/Desktop/original_mps.pcd')
labeled_pcd_file = open('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SGD_mps.pcd', 'w')
label_file = np.loadtxt('/media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/Label.txt')
# colors = [[] for i in range(num_kinds)]
# for clr in range(num_kinds):
#     for i in range(3):
#         colors[clr].append(random.randint(0, 255))
#colors=[[255,16,16],[16,255,16],[16,16,255],[16,255,255],[255,16,255],[255,255,16]]
colors=[[16,255,16],[16,255,16],[16,255,16],[16,255,16],[16,255,16],[255,16,16]]

line_num = 1
for pcd_line in original_pcd_file:
    print(line_num)
    print(pcd_line)
    if line_num < 12:
        labeled_pcd_file.writelines(pcd_line)
    else:
        idx=int(label_file[line_num - 12])
        r = hex(colors[idx][0])
        g = hex(colors[idx][1])
        b = hex(colors[idx][2])
        print(r,g,b)
        color_hex = '0x00' + str(r)[2:] + str(g)[2:] + str(b)[2:]
        color_int = int(color_hex, 16)
        pcd_line = pcd_line.split(' ')
        pcd_line = pcd_line[0] + ' ' + pcd_line[1] + ' ' + pcd_line[2] + ' ' + str(color_int) + '\n'
        labeled_pcd_file.writelines(pcd_line)
    line_num += 1
original_pcd_file.close()
labeled_pcd_file.close()
os.system('pcl_viewer /media/psf/Home/SLAMCode/0002code/core/algorithm_vehicle/vehicle/offlineSLAM/RESULTS/SGD_mps.pcd')
