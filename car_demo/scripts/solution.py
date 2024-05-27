#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
import time

import os
import numpy as np

from math import pi

def preprocess(img):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img *= 255
        img = img.astype(np.uint8)
    return img

def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]+200).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel

def to_polar_coords(x_pixel, y_pixel):
    dists = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dists, angles

def threshold(img, thresh, error):
    thresholded = np.zeros((img.shape[0], img.shape[1]))
    indices = (abs(img[:,:,0] - img[:,:,1]) < error) & \
              (abs(img[:,:,1] - img[:,:,2]) < error) & \
              (abs(img[:,:,2] - img[:,:,0]) < error) & \
              (img[:,:,0] > thresh) & \
              (img[:,:,1] > thresh) & \
              (img[:,:,2] > thresh)
    thresholded[indices] = 255
    return thresholded

def pipeline(img):
    img = img.copy()
    img = preprocess(img)
    height, width, _ = img.shape

    # plt.figure(figsize=(12, 5))
    # plt.suptitle(path.split('/')[1])

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.tight_layout()
    
    img = threshold(img, 135, 10)

    # plt.subplot(122)
    # plt.imshow(img, cmap='gray')
    # plt.tight_layout()

    x_pix, y_pix = rover_coords(img)
    dists, angles = to_polar_coords(x_pix, y_pix)
    mean_dist = np.mean(x_pix)
    mean_angle = np.mean(angles)

    return mean_dist, mean_angle

class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

class SolutionNode(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        ### Subscriber to the image topic
        self.subscriber = self.create_subscription(Image,"/prius/front_camera/image_raw",self.callback,10)
        ### Publisher to the control topic
        self.publisher = self.create_publisher(Control, "/prius/control", qos_profile=10)
        self.fps_counter = FPSCounter()
        
        self.num_frames = 0

        self.bridge = CvBridge()
        self.command = Control()
        self.ANGLE_THRESHOLD = 0.1 #TODO
    
    def draw_fps(self, img):
        self.fps_counter.step()
        fps = self.fps_counter.get_fps()
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img
    
    def draw_control_data(self, img, throttle, brake, steer):
        cv2.putText(
            img,
            f"Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img
    
    def draw_readings(self, img, dist, angle):
        cv2.putText(
            img,
            f"Distance: {dist:.2f}, Angle: {angle:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return img
    
    # def get_acceleration(self, steering_angle):
    #     steering_angle = 1/abs(steering_angle)
    #     OLD_MIN = 0
    #     OLD_MAX = 200

    #     NEW_MIN = -1.0
    #     NEW_MAX = 1.0

    #     return (steering_angle-OLD_MIN)/(OLD_MAX-OLD_MIN) * (NEW_MAX-NEW_MIN) + NEW_MIN
    
    def get_speed_control(self, angle):
        # acceleration = self.get_acceleration(angle)
        throttle = 0.65
        # brake = min(0.0,acceleration)
        steering_angle = abs(self.get_steering_angle(angle))
        # steering_angle = abs(angle)
        if steering_angle<0.12:
            throttle = 0.75
        
        if steering_angle < 0.10:
            throttle =  1.0

        if steering_angle>0.7:
            throttle = 1.0
        OLD_MIN = 0
        OLD_MAX = 1.5

        NEW_MIN = 0.0
        NEW_MAX = 1.2
        if (self.num_frames<=150):
            return 1.0,0.0
        brake =  (steering_angle-OLD_MIN)/(OLD_MAX-OLD_MIN) * (NEW_MAX-NEW_MIN) + NEW_MIN
        # brake = 0.0
        return throttle, brake
    
    def get_steering_angle(self, angle):
        OLD_MIN = -1.5
        OLD_MAX = 1.5

        NEW_MIN = -2
        NEW_MAX = 2
        return (angle-OLD_MIN)/(OLD_MAX-OLD_MIN) * (NEW_MAX-NEW_MIN) + NEW_MIN

        
    def callback(self,msg:Image):
        self.num_frames+=1
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # if self.num_frames % 10 == 0:
        #     self.save_image(cv_image)   

        cv_image = self.draw_fps(cv_image)
        
        mean_dist, mean_angle = pipeline(cv_image)

        # self.command.throttle = self.getThrottle(mean_dist)*0.7

        # self.command.steer = mean_angle * 180 / pi *0.7
        # self.command.throttle = 0.2
        # # self.command.throttle = 1.0
        # # self.command.steer = mean_angle / self.get_normalized_distance(mean_dist)
        # if abs(self.get_steering_angle(mean_angle))>0.7:
        #     self.command.brake = 1.0
        
        self.command.steer = self.get_steering_angle(mean_angle)
        self.command.throttle, self.command.brake = self.get_speed_control(mean_angle)

        self.command.steer*=1.25

        cv_image = self.draw_readings(cv_image, mean_dist, mean_angle)
        cv_image = self.draw_control_data(cv_image,self.command.throttle,self.command.brake,self.command.steer)
        self.publisher.publish(self.command)
        #### show image
        cv2.imshow("prius_front",cv_image)
        cv2.waitKey(5)

    def save_image(self, image:Image):
        path  = os.path.join(os.getcwd(),f"{self.num_frames//10}.jpg")
        if not cv2.imwrite(path,image):
            raise Exception("Couldn't save image")

def main():
    rclpy.init()
    node = SolutionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()