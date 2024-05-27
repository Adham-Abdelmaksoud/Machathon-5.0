import json
import matplotlib.pyplot as plt

def read_data(batch_size):
    annotations_path = 'DeepLearning/annotations.json'
    imgs_dir = 'final_imgs'

    with open(annotations_path, 'r') as f:
        annotations = json.loads(f.read())
    
    imgs = []
    angles = []

    counter = 0
    imgs_batch = []
    angles_batch = []

    for img_name in annotations:
        imgs_batch.append(plt.imread(f'{imgs_dir}/{img_name}'))
        angles_batch.append(annotations[img_name])
        counter += 1

        if counter == batch_size:
            counter = 0
            imgs.append(imgs_batch)
            angles.append(angles_batch)
            imgs_batch = []
            angles_batch = []
    
    if counter != 0:
        imgs.append(imgs_batch)
        angles.append(angles_batch)

    return imgs, angles