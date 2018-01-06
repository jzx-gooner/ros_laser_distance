#coding=utf-8
from __future__ import division
from numpy import *
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
#细化函数，提取激光骨架
def Thin(image,array):
    h= image.shape[0]
    w = image.shape[1]
    iThin=image.copy()
    for i in range(h):
        for j in range(w):
            if image[i,j] == 0:
                a = [1]*9
                for k in range(3):
                    for l in range(3):
                        if -1<(i-1+k)<h and -1<(j-1+l)<w and iThin[i-1+k,j-1+l]==0:
                            a[k*3+l] = 0
                sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                iThin[i,j] = array[sum]*255
    return iThin
#根据八个点的情况判断该点能否舍去
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

def read_picture():
    img=cv2.imread("new.jpg")
    # 把 BGR 转为 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 风机叶片中激光范围
    lower_red1 = np.array([0,43,46])
    lower_red2 = np.array([156,43,46])
    upper_red1 = np.array([10,255,255])
    upper_red2 = np.array([180,255,255])
    # 获得红色区域的mask
    mask = cv2.inRange(hsv, lower_red1, upper_red1)|cv2.inRange(hsv, lower_red2, upper_red2)
    # 和原始图片进行and操作，获得红色区域
    res = cv2.bitwise_and(img,img, mask= mask)
    #灰度化
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #二值化
    gray_src = cv2.bitwise_not(gray)
    binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15, -2)
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((img.shape[1]//16), 1), (-1, -1))
    # 提取水平线
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((img.shape[1]//16), 1), (-1, -1))
    # 提取垂直线
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (img.shape[0]//16)), (-1, -1))
    #垂直
    vline_dst = cv2.morphologyEx(binary_src, cv2.MORPH_OPEN, vline)
    vline_dst = cv2.bitwise_not(vline_dst)
    #水平
    hline_dst = cv2.morphologyEx(binary_src, cv2.MORPH_OPEN, hline)
    hline_dst = cv2.bitwise_not(hline_dst)
    #提取骨架
    vline_dst_thin=Thin(vline_dst,array)
    hline_dst_thin=Thin(hline_dst,array)
    #读取激光点的
    height = vline_dst_thin.shape[0]
    width =vline_dst_thin.shape[1]
    v_laser_dot=[]
    h_laser_dot=[]
    for i in range(height):
        for j in range(width):
            #print(vline_dst_thin[i,j])
            if vline_dst_thin[i,j]!=255:
                v_laser_dot.append([j,i])
            if vline_dst_thin[i,j]!=255:
                h_laser_dot.append([j,i])
    #print(v_laser_dot)
    #print(h_laser_dot)
    # num=(len(v_laser_dot))
    plt.subplot(221)
    plt.imshow(gray_src)
    plt.subplot(222)
    plt.imshow(vline)
    plt.subplot(223)
    plt.imshow(vline_dst)
    plt.subplot(224)
    plt.imshow(vline_dst_thin)
    plt.show()
    a=0
    b=0
    laser_dot=v_laser_dot+h_laser_dot
    print(len(v_laser_dot))
    num=len(laser_dot)
    for i in laser_dot:
        a+=i[0]
        b+=i[1]
    laser_x=a/num
    laser_y=b/num
    return num,v_laser_dot,h_laser_dot,laser_x,laser_y
if __name__ == '__main__':
    num,v_laser_dot,h_laser_dot,laser_x,laser_y=read_picture()
    print('the num of the laser dot is',num)
    print('the average of the position is',[laser_x,laser_y])
