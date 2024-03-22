import os
import logging
import cv2
import numpy as np
import glob
import random
from datetime import datetime
from sklearn.model_selection import train_test_split

def get_files(dir, format="jpg", n=-1):
    files = glob.glob(os.path.join(dir, f"*.{format}"))
    if n > 0:
        return random.sample(files, min(n, len(files)))
    return files

def n_partition_data(real_dir,fake_dir,n,p,real_scale,fake_to_real_ratio,format="jpg",augmentation=False,augment_ratio=0.5,resize=0):
    images = []
    labels = []
    
    real_files = get_files(real_dir, format, int(n * real_scale))
    fake_files = get_files(fake_dir, format)[: len(real_files) * fake_to_real_ratio]
    logging.info(f"Loaded {len(real_files)} real images and {len(fake_files)} fake images")
    if n >= 1:
        real_files = real_files[p * (n - 1) : p * n]
        fake_files = fake_files[p * (n - 1) : p * n]
        print(real_files)
        logging.info(f"Partitioned data into {n} parts, using part {p}")
    for file in real_files:
        img = cv2.imread(file)
        if resize!=0:
            img = cv2.resize(img, (resize, resize))
        images.append(img)
        labels.append(0)
    
    for file in fake_files:
        img = cv2.imread(file)
        if resize!=0:
            img = cv2.resize(img, (resize, resize))
        images.append(img)
        labels.append(1)
    logging.info(f"Loaded {len(images)} images and {len(labels)} labels")
    if augmentation:
        logging.info(f"Applying data augmentation with ratio {augment_ratio}")
        aug_images, aug_labels = data_augmentation(images,labels,augment_ratio)
        images.extend(aug_images)
        labels.extend(aug_labels)
        logging.info(f"Size after augmentation: {len(images)}")
        
    return np.array(images), np.array(labels)
    

def load_data(real_dir,fake_dir, real_limit, fake_limit_scale,augmentation=False, augment_ratio=0.5,resize=224):
    images = []
    labels = []
    
    real_files = get_files(real_dir, "jpg", real_limit)
    if fake_limit_scale>0:
        fake_files = get_files(fake_dir, "jpg", -1)[:int(fake_limit_scale*len(real_files))]
    else:
        fake_files = get_files(fake_dir, "jpg", -1)

    for file in real_files:
        img = cv2.imread(file)
        if resize!=224:
            img = cv2.resize(img, (resize, resize))
        images.append(img)
        labels.append(0)
    rl=len(images)
    logging.info(f"{rl} number of real images processed")
    
    for file in fake_files:
        img = cv2.imread(file)
        if resize!=224:
            img = cv2.resize(img, (resize, resize))
        images.append(img)
        labels.append(1)
    fl = len(images)
    logging.info(f"{fl-rl} number of fake images processed")

    if resize!=224:
        logging.info(f"Images resized to {resize}x{resize} pixels")

    logging.info(f"Loaded {len(images)} images and {len(labels)} labels")
    if augmentation:
        logging.info(f"Applying data augmentation with ratio {augment_ratio}")
        aug_images, aug_labels = data_augmentation(images,labels,augment_ratio)
        images.extend(aug_images)
        labels.extend(aug_labels)
        logging.info(f"Size after augmentation: {len(images)}")
        
    return np.array(images), np.array(labels)

def load_paths(dir):
    return list(map(lambda x: os.path.join(dir, x), os.listdir(dir)))

def split_data(X, y, val_split):
    return train_test_split(X, y, test_size=val_split, random_state=42)

def data_augmentation(images,labels,ratio=0.5):
    n = len(images)
    indices = random.sample(range(n),int(n*ratio))
    augmented_images = []
    augmented_labels = []
    for i in indices:
        img = images[i]
        label = labels[i]
        if i%2==0:
            img = cv2.flip(img,1)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(img)
        augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)
    