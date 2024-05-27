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


def to_polar_coords(x_pixel, y_pixel):
    dists = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dists, angles


def threshold_red(img, error):
    thresholded = np.zeros((img.shape[0], img.shape[1]))
    indices = ((cv2.subtract(img[:,:,0], img[:,:,1]) / img[:,:,0]) > error) & \
              ((cv2.subtract(img[:,:,0], img[:,:,2]) / img[:,:,0]) > error)
    thresholded[indices] = 255
    return thresholded


def pipeline(img):
    NUM_IMGS = 2

    img = img.copy()
    img = preprocess(img)
    height, width, channels = img.shape

    plt.figure(figsize=(12, 5))

    plt.subplot(int(f'1{NUM_IMGS}1'))
    plt.imshow(img)
    plt.tight_layout()
    
    # img = threshold_red(img, 50).astype(np.uint8)
    img = threshold_red(img, 0.15).astype(np.uint8)

    plt.subplot(int(f'1{NUM_IMGS}2'))
    plt.imshow(img, cmap='gray')
    plt.tight_layout()

    img1, img2 = img[:,:width//2], img[:,width//2:]

    x_pix1, y_pix1 = rover_coords(img1)
    x_pix2, y_pix2 = rover_coords(img2)

    try:
        coeffs1 = np.polyfit(x_pix1, y_pix1, 1)
        coeffs2 = np.polyfit(x_pix2, y_pix2, 1)

        angle1 = np.arctan(coeffs1[0])
        angle2 = np.arctan(coeffs2[0])
        if abs(angle1) > 1.3 or abs(angle2) > 1.3:
            if angle1*angle2 < 0:
                if abs(angle1) > abs(angle2):
                    angle1 *= -1
                else:
                    angle2 *= -1
                
        # if abs(angle1) > abs(angle2):
        #     mean_angle = angle1*0.65 + angle2*0.35
        # else:
        #     mean_angle = angle1*0.35 + angle2*0.65 
        # mean_angle = (angle1 + angle2) / 2

        factor = 2
        ratio = (len(x_pix1) / len(x_pix2)) * factor
        mean_angle = angle1*ratio + angle2*(1-ratio)
        # if len(x_pix1) > len(x_pix2):
        #     mean_angle = angle1*0.65 + angle2*0.35
        # else:
        #     mean_angle = angle1*0.35 + angle2*0.65
    except:
        angle1 = 0
        angle2 = 0
        mean_angle = 0

    arrow_length = 250
    x_arrow = arrow_length * np.cos(mean_angle)
    y_arrow = arrow_length * np.sin(mean_angle)
    plt.arrow(width//2-3, height-3, -y_arrow, -x_arrow, color='red', zorder=2, head_width=15, width=2)

    return mean_angle
    

    


if __name__ == '__main__':
    input_dir_path = 'all_imgs_real'

    imgs_names = os.listdir(input_dir_path)
    imgs_names.sort(key=lambda x: len(x))

    from tqdm import tqdm

    for img_name in tqdm(imgs_names, disable=True):
        img = plt.imread(f'{input_dir_path}/{img_name}')

        angle = pipeline(img)

        plt.show()
        # plt.savefig(f'outputs_real/{img_name}')
        plt.close()