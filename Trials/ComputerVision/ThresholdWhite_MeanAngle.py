import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from time import time


enable = False
def measure_time(func):
    def inner(*args):
        time1 = time()
        value = func(*args)
        time2 = time()
        if enable:
            print(f'Elapsed time for {func.__name__}(): {time2 - time1} seconds')
        return value
    return inner

@measure_time
def preprocess(img):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img *= 255
        img = img.astype(np.uint8)
    return img

@measure_time
def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel

@measure_time
def to_polar_coords(x_pixel, y_pixel):
    dists = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dists, angles

@measure_time
def threshold_white_degrees(img, white_thresh, error):
    thresholded = np.zeros((img.shape[0], img.shape[1]))
    indices = (cv2.absdiff(img[:,:,0], img[:,:,1]) < error) & \
              (cv2.absdiff(img[:,:,1], img[:,:,2]) < error) & \
              (cv2.absdiff(img[:,:,2], img[:,:,0]) < error) & \
              (img[:,:,0] > white_thresh) & \
              (img[:,:,1] > white_thresh) & \
              (img[:,:,2] > white_thresh)
    thresholded[indices] = 255
    return thresholded

@measure_time
def pipeline(img):
    img = img.copy()
    img = preprocess(img)
    height, width, channels = img.shape

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.imshow(img)
    plt.tight_layout()
    
    img = threshold_white_degrees(img, 135, 5)

    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.tight_layout()

    x_pix, y_pix = rover_coords(img)
    dists, angles = to_polar_coords(x_pix, y_pix)
    mean_dist = np.mean(dists)
    mean_angle = np.mean(angles)

    x_arrow = mean_dist * np.cos(mean_angle)
    y_arrow = mean_dist * np.sin(mean_angle)
    plt.arrow(width//2-3, height-3, -y_arrow, -x_arrow, color='red', zorder=2, head_width=15, width=2)

    return mean_dist, mean_angle
    

    


if __name__ == '__main__':
    input_dir_path = 'final_imgs'

    imgs_names = os.listdir(input_dir_path)
    imgs_names.sort(key=lambda x: len(x))

    from tqdm import tqdm

    for img_name in tqdm(imgs_names, disable=True):
        img = plt.imread(f'{input_dir_path}/{img_name}')

        dist, angle = pipeline(img)

        plt.show()
        # plt.savefig(f'outputs/{img_name}')
        plt.close()