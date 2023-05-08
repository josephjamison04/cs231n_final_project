import cv2
import numpy as np
import os


path = '/Users/josephjamison/Downloads/data/images/train/s/'
folder_list = [i for i in os.listdir(path) if not i.startswith('.')]

images = []
labels = []

failed_folders = []

for folder in folder_list:
    
    dir_list = os.listdir(os.path.join(path, folder))
    for file in dir_list:
        try:
            im = cv2.imread(os.path.join(path, folder, file)).transpose((2, 0, 1))
            images.append(im)
            labels.append(folder)
        except:
            print("Warning: One or more files could not be read")
    

images = np.array(images)
labels = np.array(labels)

print(images.shape)
print(labels.shape)
print(np.unique(labels))
