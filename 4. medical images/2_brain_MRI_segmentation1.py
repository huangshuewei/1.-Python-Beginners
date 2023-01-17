# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:38:53 2023

@author: ASUS

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
"""

import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from keras import preprocessing

x_train = []
y_mask = []

train_files = []
mask_files = glob('data/kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))

# Data visulisation
rows,cols = 1,4
fig = plt.figure(figsize=(6,8))
for i in range(1,rows*cols+1):
    plt.subplot(rows+1,cols,i)
    msk_path = mask_files[i]
    msk = cv2.imread(msk_path, 0)
    plt.title("Masks")
    plt.axis('off')
    plt.imshow(msk, cmap='gray')
    fig.add_subplot(rows,cols,i)
    img_path = train_files[i]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  
    plt.title("Images")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

rows,cols = 1,4
fig = plt.figure(figsize=(6,8))
for i in range(1,rows*cols+1):
    plt.subplot(rows,cols,i)
    img_path = train_files[i]
    msk_path = mask_files[i]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    msk = cv2.imread(msk_path, 0)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.4, cmap='gray')
    plt.axis('off')
    plt.title("Image&Mask")
plt.show()

# Load data
IMG_SIZE = 256
IMG_CHANNELS = 3

# Data preparation
from sklearn.model_selection import train_test_split
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})

def tumor_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0 : return 1
    else: return 0

# The brain has a cancer or not
df["diagnosis"] = df["mask"].apply(lambda m: tumor_diagnosis(m))

# Separate data
df_train, df_test = train_test_split(df,test_size = 0.1, random_state = 42)
df_train, df_val = train_test_split(df_train,test_size = 0.2, random_state = 42)
print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)

# Data generation / Image augmentation
# From: https://github.com/zhixuhao/unet/blob/master/data.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)
##############################################################################
BATCH_SIZE = 16
train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE,
                                train_generator_args,
                                target_size=(IMG_SIZE, IMG_SIZE))
    
val_gen = train_generator(df_val, BATCH_SIZE,
                                dict(),
                                target_size=(IMG_SIZE, IMG_SIZE))

test_gen = train_generator(df_test, BATCH_SIZE,
                                dict(),
                                target_size=(IMG_SIZE, IMG_SIZE))

# import segmentation_models as sm
import tensorflow as tf

# Build the model
inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

# Contract path
# s = tf.keras.layers.Lambda(lambda x: x/255.)(inputs)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_unet_cus.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
            checkpointer]

history = model.fit_generator(train_gen,
                              steps_per_epoch=len(df_train) / BATCH_SIZE,
                              epochs=100, 
                              callbacks=callbacks,
                              validation_data = val_gen,
                              validation_steps=len(df_val) / BATCH_SIZE)

# Evaluating model performance
results = model.evaluate(test_gen, steps=len(df_test) / BATCH_SIZE)
print("Test lost: ",results[0])
print("Test Acc: ",results[1])

import random

rows,cols = 4,4
img_list = random.sample(range(0, len(df_test)), rows*cols)
test_img_list = list(df_test['filename'])
test_msk_list = list(df_test['mask'])

fig = plt.figure(figsize=(8,8))
for i, i_img in enumerate(img_list):
    plt.subplot(rows,cols,i+1)
    img_path = test_img_list[i_img]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    msk_path = test_msk_list[i_img]
    msk = cv2.imread(msk_path, 0)
    plt.title("Images&Mask")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.imshow(msk,alpha=0.4, cmap='gray')
plt.show()

# prediction
fig = plt.figure(figsize=(8,8))
for i, i_img in enumerate(img_list):
    plt.subplot(rows,cols,i+1)
    img_path = test_img_list[i_img]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    img = np.expand_dims(img,axis=0)
    predict_mask = model.predict(img)
    predict_mask[predict_mask>0.5] = 1
    predict_mask[predict_mask<=0.5] = 0
    plt.title("Imgs&PreMasks")
    plt.axis('off')
    plt.imshow(img[0], cmap='gray')
    plt.imshow(predict_mask[0],alpha=0.4 , cmap='gray')
plt.show()
