#!/usr/bin/python
# coding:utf-8

import numpy as np
import os

PGL_PATH = "/home/test1234/Desktop/Paint_and_Cameracenter"
PLY_PATH = PGL_PATH
pgl_list = "/home/test1234/Desktop/Paint_and_Cameracenter/pgl_list.txt"
ply_head=open("/home/test1234/Desktop/Paint_and_Cameracenter/plyhead.ply",'r+')
plyheadlist=ply_head.readlines()

file = open(pgl_list)
stringlist = file.readlines()

for string in stringlist:
    num = string.rfind(".pgl")
    valstring = string[1: num]

    pgldir = PGL_PATH + valstring + ".pgl"
    plydir = PLY_PATH + valstring + ".ply"

    pgl_file = open(pgldir)
    ply_file = open(plydir,'w+')
    ply_file.writelines(plyheadlist)

    line_num = 1
    for pgl_line in pgl_file:
        if line_num == 1:
            line_num +=1
        else:
            pgl_line = pgl_line.split(', ')
            ply_line=pgl_line[3] + ' ' + pgl_line[4] + ' ' + pgl_line[5] + ' 192 192 192 255' + '\n'
            ply_file.writelines(ply_line)
            line_num += 1
    ply_file.writelines("0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 "+str(line_num-2)+" 1 0 0")
    pgl_file.close()
    ply_file.close()
    text1 = "element vertex " + str(line_num-2) + "\n"
    f=open(plydir,'r+')
    flist=f.readlines()
    flist[3]=text1
    f=open(plydir,'w+')
    f.writelines(flist)
    f.close()
ply_head.close()




