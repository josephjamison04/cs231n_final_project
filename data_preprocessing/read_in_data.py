import cv2
import numpy as np
import os
import pandas as pd


def process_images():
    path = '/Users/josephjamison/Downloads/CS231n_temp/data/images/train/'
    folder_list = [i for i in os.listdir(path) if not i.startswith('.')]

    total_images_processed = 0
    image_idx_pairs = []
    images = []
    labels = []

    for first_letter in folder_list:
        letter_list = [i for i in os.listdir(os.path.join(path, first_letter)) if not i.startswith('.')]
        for folder in letter_list:
            dir_list = [i for i in os.listdir(os.path.join(path, first_letter, folder)) if not i.startswith('.')]
            for file in dir_list:
                try:
                    im = cv2.imread(os.path.join(path, first_letter, folder, file)).transpose((2, 0, 1))
                    images.append(im)
                    labels.append(folder)
                    image_idx_pairs.append([folder + file,  total_images_processed])
                    total_images_processed += 1
                except:
                    sub_dir_list = [i for i in os.listdir(os.path.join(path, first_letter, folder, file)) 
                                                                                    if not i.startswith('.')]
                    try: 
                        for file2 in sub_dir_list:
                            im = cv2.imread(os.path.join(path, first_letter, folder, file, file2)).transpose((2, 0, 1))
                            images.append(im)
                            labels.append(folder + "_" + file)
                            image_idx_pairs.append([folder + "_" + file + file2,  total_images_processed])
                            total_images_processed += 1
                    except:
                        print("Warning: One or more folder / file could not be read")
            

        # images = np.array(images)
        # labels = np.array(labels)

        # # print(images.shape)
        # # print(labels.shape)
        # print(f"Unique labels from folder of letter: {first_letter}")
        # print(np.unique(labels))
        # total_images_processed += labels.shape[0]
        print(f"Just finished folder: {first_letter}. Total number of images processed: {total_images_processed}")

    images = np.array(images)
    labels = np.array(labels)
    image_idx_pairs = np.array(image_idx_pairs)
    N = images.shape[0]
    assert N == labels.shape[0]
    assert N == image_idx_pairs.shape[0]
    print("images.shape: ", images.shape)

    # Assign unique integer to each label 
    label_names = []
    counter = 0
    for i in np.unique(labels):
        label_names.append([i, counter])
        counter += 1
    # print("Label_names: ", label_names)

    # Replace labels with integers
    for i in label_names:
        labels[labels==i[0]] = i[1]
    

    # Write array of image to index pairs to csv
    image_pairs_DF = pd.DataFrame(image_idx_pairs)
    image_pairs_DF.to_csv("image_idx_pairs.csv", index=False)

    # Write dictionary of label strings to integers to csv
    label_names_DF = pd.DataFrame(label_names)
    label_names_DF.to_csv("label_names.csv", index=False)
    
    # Write images and labels to csv 
    labels_DF = pd.DataFrame(labels)
    labels_DF.to_pickle("small_images_labels.pkl")
    image_DF = pd.DataFrame(images.reshape(N, -1))
    # image_DF.to_pickle("small_images_all_data.pkl")


def main():
    process_images()

if __name__=="__main__":
    main()