import os
import cv2
import matplotlib.pyplot as plt

img = plt.imread('outputs/1.jpg')
h, w, _ = img.shape
video_writer = cv2.VideoWriter('output_video.avi', 0, 20, (w,h))

img_names = os.listdir('outputs')
img_names.sort(key=lambda x: len(x))

for img_name in img_names:
    img = plt.imread(os.path.join('outputs', img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    video_writer.write(img)

video_writer.release()