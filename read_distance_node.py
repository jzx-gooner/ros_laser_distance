#!/usr/bin/env python
# coding: utf-8

import sys, rospy, cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
#from __future__ import print_function
# class
class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("laser_distance", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/fpv_camera_ros/image_raw", Image, self.callback)
    #
    def callback(self,data):
        # imgmsg_to_cv2
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows > 60:
            cv2.circle(cv_image, (50,50), 10, 255)
            #cv2.putText(cv_image, "Hello World", (30,30),(0,255,0))
        cv2.imshow("Image windows", cv_image)
        cv2.waitKey(3)
        #把BGR转换成HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 风机叶片中激光范围
        lower_red = np.array([0,43,46])
        upper_red = np.array([10,255,255])
        # 获得红色区域的mask
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(cv_image,cv_image, mask= mask)
        height = res.shape[0]
        width =res.shape[1]
        laser_dot=[]
        for i in range(height):
            for j in range(width):
                if res[i,j][0]!=0:
                    laser_dot.append([i,j])
        num=(len(laser_dot))
        a=0
        b=0
        for i in laser_dot:
            a+=i[0]
            b+=i[1]
        laser_x=a/num
        laser_y=b/num
        print(laser_x,laser_y,num)

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
