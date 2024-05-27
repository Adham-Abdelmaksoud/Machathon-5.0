import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def preprocess(img):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img *= 255
        img = img.astype(np.uint8)
    return img


def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel


def pipeline(img):
    NUM_IMGS = 3

    img = img.copy()
    img = preprocess(img)
    height, width, channels = img.shape

    plt.figure(figsize=(12, 5))
    plt.suptitle(path.split('/')[1])

    plt.subplot(int(f'1{NUM_IMGS}1'))
    plt.imshow(img)
    plt.tight_layout()
    
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    plt.subplot(int(f'1{NUM_IMGS}2'))
    plt.imshow(img, cmap='gray')
    plt.tight_layout()

    img = cv2.Canny(img, 20, 130)

    h_kernel = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ]).astype(np.uint8)
    img = cv2.erode(img, h_kernel, iterations=1)
    img = cv2.dilate(img, h_kernel, iterations=1)
    img = img[10:,:]
    height -= 10

    img1, img2 = img[:,:width//2], img[:,width//2:]

    x_pix1, y_pix1 = rover_coords(img1)
    coeffs1 = np.polyfit(x_pix1, y_pix1, 1)

    x_pix2, y_pix2 = rover_coords(img2)
    coeffs2 = np.polyfit(x_pix2, y_pix2, 1)

    angle1 = np.arctan(coeffs1[0])
    angle2 = np.arctan(coeffs2[0])
    if abs(angle1) > 1.5 or abs(angle2) > 1.5:
        if angle1*angle2 < 0:
            if abs(angle1) > abs(angle2):
                angle1 *= -1
            else:
                angle2 *= -1
            
    if abs(angle1) > abs(angle2):
        mean_angle = angle1*0.6 + angle2*0.4
    else:
        mean_angle = angle1*0.4 + angle2*0.6

    plt.subplot(int(f'1{NUM_IMGS}3'))
    plt.imshow(img, cmap='gray')
    plt.tight_layout()

    arrow_length = 250
    x_arrow = arrow_length * np.cos(mean_angle)
    y_arrow = arrow_length * np.sin(mean_angle)
    plt.arrow(width//2-3, height-3, -y_arrow, -x_arrow, color='red', zorder=2, head_width=15, width=2)

    return mean_angle
    


if __name__ == '__main__':
    input_dir_path = 'saturation_imgs'

    imgs_names = os.listdir(input_dir_path)
    # imgs_names.sort(key=lambda x: len(x))

    for img_name in imgs_names:
        path = f'{input_dir_path}/{img_name}'
        img = plt.imread(path)

        angle = pipeline(img)

        plt.show()        
        plt.close()