import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from time import time
from collections import defaultdict



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
def reverse_rover_coords(img, x_pixel, y_pixel):
    ypos = -x_pixel + img.shape[0]
    xpos = -y_pixel + img.shape[1]/2
    return xpos, ypos

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
def connected_components(img):
    analysis = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    totalLabels, label_ids, values, centroid = analysis
    output = np.zeros(img.shape, dtype="uint8")
    max_area = 0

    for i in range(1, totalLabels): 
        area = values[i, cv2.CC_STAT_AREA]
        componentMask = (label_ids == i).astype(np.uint8) * 255

        if len(componentMask[img.shape[0]-5, :].nonzero()[0]) == 0:
            continue
        if area > max_area:
            max_area = area
            output = componentMask

    return output

@measure_time
def calculate_poly(x, coeffs):
    output = 0
    for i in range(len(coeffs)):
        output += coeffs[i] * x**(len(coeffs)-1-i)
    return output

@measure_time
def get_average_pix(x_pix, y_pix):
    max_pixs = dict()
    min_pixs = dict()

    max_pixs = defaultdict(lambda: -1000)
    min_pixs = defaultdict(lambda: 1000)
    for x, y in zip(x_pix, y_pix):
        max_pixs[x] = max(max_pixs[x], y)
        min_pixs[x] = min(min_pixs[x], y)

    avg_x = [0]*100 + [30]*120
    avg_y = [0]*220
    for key in sorted(max_pixs.keys()):
        avg_x.append(key)
        avg_y.append((max_pixs[key] + min_pixs[key]) // 2)
    avg_x.extend([avg_x[-1]]*50)
    avg_y.extend([avg_y[-1]]*50)

    return avg_x, avg_y


@measure_time
def pipeline(img):
    img = img.copy()
    img = preprocess(img)
    height, width, channels = img.shape

    plt.figure(figsize=(12, 5))
    plt.suptitle(path.split('/')[1])

    plt.subplot(121)
    plt.imshow(img)
    plt.tight_layout()
    
    img = threshold_white_degrees(img, 135, 10).astype(np.uint8)

    new_img = cv2.GaussianBlur(img, (3,3), 0)
    new_img = connected_components(img)
    x_pix, y_pix = rover_coords(new_img)

    avg_x, avg_y = get_average_pix(x_pix, y_pix)

    coeffs = np.polyfit(avg_x, avg_y, 3)

    x = 70
    y = calculate_poly(x, coeffs)

    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    val1, val2 = reverse_rover_coords(img, x_pix, calculate_poly(x_pix, coeffs))
    plt.plot(val1, val2, 'g')
    plt.arrow(width//2, height-2, -y, -x, color='red', zorder=2, head_width=10, width=2)
    plt.tight_layout()

    dist, angle = to_polar_coords(50, calculate_poly(50, coeffs))
    return dist, angle
    


if __name__ == '__main__':
    input_dir_path = 'test_imgs'

    imgs_names = os.listdir(input_dir_path)
    imgs_names.sort(key=lambda x: len(x))

    for img_name in imgs_names:
        path = f'{input_dir_path}/{img_name}'
        img = plt.imread(path)

        dist, angle = pipeline(img)

        plt.show()        
        plt.close()