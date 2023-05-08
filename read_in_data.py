import cv2
import numpy as np
import os


path = '/Users/josephjamison/Downloads/data/images/train/'
folder_list = [i for i in os.listdir(path) if not i.startswith('.')]

total_images_processed = 0

for first_letter in folder_list:
    images = []
    labels = []
    letter_list = [i for i in os.listdir(os.path.join(path, first_letter)) if not i.startswith('.')]

    for folder in letter_list:
        dir_list = [i for i in os.listdir(os.path.join(path, first_letter, folder)) if not i.startswith('.')]
        
        for file in dir_list:
            try:
                im = cv2.imread(os.path.join(path, first_letter, folder, file)).transpose((2, 0, 1))
                images.append(im)
                labels.append(folder)
            except:
                sub_dir_list = [i for i in os.listdir(os.path.join(path, first_letter, folder, file)) 
                                                                                if not i.startswith('.')]
                try: 
                    for file2 in sub_dir_list:
                        im = cv2.imread(os.path.join(path, first_letter, folder, file, file2)).transpose((2, 0, 1))
                        images.append(im)
                        labels.append(folder + "_" + file)
                except:
                    print("Warning: One or more folder / file could not be read")
        

    images = np.array(images)
    labels = np.array(labels)

    # print(images.shape)
    # print(labels.shape)
    print(f"Unique labels from folder of letter: {first_letter}")
    print(np.unique(labels))
    total_images_processed += labels.shape[0]

print(f"Total number of images processed: {total_images_processed}")


