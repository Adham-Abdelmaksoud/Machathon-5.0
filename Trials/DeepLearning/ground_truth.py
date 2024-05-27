import cv2
import json
import os
import numpy as np

base_dir = '.'
file_path = f'{base_dir}/annotations.json'
try:
    with open(file_path, 'r') as f:
        annotations = json.loads(f.read())
except:
    annotations = dict()
    with open(file_path, 'w') as f:
        pass

class BoundingBoxWidget(object):
    def __init__(self, img_path):
        self.original_image = cv2.imread(img_path)
        self.height, self.width, self.channels = self.original_image.shape
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        self.coords = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.coords = (x,y)
            self.origin = (self.width//2,self.height)
            self.angle = np.arctan((self.origin[0]-self.coords[0]) / (self.origin[1]-self.coords[1]))

        elif event == cv2.EVENT_LBUTTONUP:
            annotations[img_name] = self.angle
            with open(file_path, 'r+') as f:
                f.write(json.dumps(annotations, indent=2))
            print(f'Angle: {self.angle}')


            cv2.line(self.clone, (self.width//2,self.height), self.coords, (36,255,12), 2)
            cv2.imshow("image", self.clone) 

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
    

def set_groundtruth(img_path):
    boundingbox_widget = BoundingBoxWidget(img_path)

    with open(file_path, 'r+') as f:
        while True:
            cv2.imshow('image', boundingbox_widget.show_image())
            input_key = cv2.waitKey(1)

            if input_key == ord('n'):
                cv2.destroyAllWindows()
                break

            if input_key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)


if __name__ == '__main__':
    dir_path = 'final_imgs'
    img_names = os.listdir(dir_path)
    img_names.sort(key=lambda x: len(x))

    for img_name in img_names:
        img_path = f'{dir_path}/{img_name}'
        set_groundtruth(img_path)