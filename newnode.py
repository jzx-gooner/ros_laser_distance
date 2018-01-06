#!/usr/bin/env python
# coding: utf-8
import sys, rospy, cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
from matplotlib import pyplot as plt
#from __future__ import print_function
# class
class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("laser_distance", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/fpv_camera_ros/image_raw", Image, self.callback)
    #计算距离
    def measure_distance(self,planeVector,planePoint,lineVector,linePoint):
        vp1 = planeVector[0]
        vp2 = planeVector[1]
        vp3 = planeVector[2]
        n1 = planePoint[0]
        n2 = planePoint[1]
        n3 = planePoint[2]
        v1 = lineVector[0]
        v2 = lineVector[1]
        v3 = lineVector[2]
        m1 = linePoint[0]
        m2 = linePoint[1]
        m3 = linePoint[2]
        resultPoint=[None]*3
        vpt = v1 * vp1 + v2 * vp2 + v3 * vp3;
        if (vpt == 0):
            resultPoint[0] = None;
            resultPoint[1] = None;
            resultPoint[2] = None;
            return None
        else:
            t = ((n1 - m1) * vp1 + (n2 - m2) * vp2 + (n3 - m3) * vp3) / vpt;
            resultPoint[0] = m1 + v1 * t;
            resultPoint[1] = m2 + v2 * t;
            resultPoint[2] = m3 + v3 * t;
            #print(resultPoint)
            distance=math.sqrt(resultPoint[0]*resultPoint[0]+resultPoint[1]*resultPoint[1]+resultPoint[2]*resultPoint[2])
            return distance

    def callback(self,data):
        # imgmsg_to_cv2
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
        #(rows, cols, channels) = cv_image.shape
        #if cols > 60 and rows > 60:
            #cv2.circle(cv_image, (50,50), 10, 255)
            #cv2.putText('there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
        img=cv_image
        cv2.imshow("Image windows", cv_image)
        cv2.waitKey(1)
        #把BGR转换成HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 风机叶片中激光范围
        lower_red1 = np.array([0,43,46])
        lower_red2 = np.array([156,43,46])
        upper_red1 = np.array([10,255,255])
        upper_red2 = np.array([180,255,255])
        # 获得红色区域的mask
        mask = cv2.inRange(hsv, lower_red1, upper_red1)|cv2.inRange(hsv, lower_red2, upper_red2)
        # 和原始图片进行and操作，获得红色区域
        res = cv2.bitwise_and(cv_image,cv_image, mask= mask)
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
        #读取激光点的
        height = vline_dst.shape[0]
        width =vline_dst.shape[1]
        v_laser_dot=[]
        h_laser_dot=[]
        for i in range(height):
            for j in range(width):
                #print(vline_dst_thin[i,j])
                if vline_dst[i,j]!=255:
                    v_laser_dot.append([j,i])
                if vline_dst[i,j]!=255:
                    h_laser_dot.append([j,i])
        a=0
        b=0
        laser_dot=v_laser_dot+h_laser_dot
        print(len(v_laser_dot))
        num=len(laser_dot)
        for i in laser_dot:
            a+=i[0]
            b+=i[1]
        laser_x=a/num if num!=0 else None
        laser_y=a/num if num!=0 else None
        print("the average location is",[laser_x,laser_y])
        pixelsize=0.006#像素尺寸 mm
        f=16 #焦距mm
        angle=87.89 #laser夹角为87.89
        planevector=[0.9993219830243762, 0.0, -0.03681812385535621]#垂直线的法向量
        linepoint=[0.0,0.0,0.0]
        planepoint=[300.00,0.0,0.0]
        for p in h_laser_dot:
            linevector=[pixelsize*(p[0]-376.0),pixelsize*(p[1]-240.0),f]
            print('the laser location is',[p[0],p[1]])
            print('the distance is',self.measure_distance(planevector,planepoint,linevector,linepoint))
        try:
            # ROSbgr8
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            ###--- OpenCV-image to ROS-Message
            #self.image_pun.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        except CvBridgeError as e:
            print e

def main(args):
    rospy.loginfo('read distance')
    ic = image_converter()
    # anonumous=True
    rospy.init_node('read_distance', anonymous=True)
    # rospy.spin()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
