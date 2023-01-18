# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:28:01 2023

the area localisation of pneumonia in lung x-ray images
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
import scipy  #Used to upsample our image

import cv2
from glob import glob
import numpy as np
import pandas as pd

# find data directory
normal_train_file = glob('data/train/normal/*')
opacity_train_file = glob('data/train/opacity/*')

normal_test_file = glob('data/test/normal/*')
opacity_test_file = glob('data/test/opacity/*')

normal_val_file = glob('data/val/normal/*')
opacity_val_file = glob('data/val/opacity/*')

val_dir = "data/val/"

# data visualisation
rows,cols = 1,4
plt.figure(figsize=(6,8))
for i_imgs in range(1, rows*cols+1):
    plt.subplot(rows,cols,i_imgs)
    normal_path = normal_train_file[i_imgs]
    normal = cv2.imread(normal_path)
    normal = cv2.cvtColor(normal,cv2.COLOR_BGR2RGB) 
    plt.title("Normal")
    plt.axis('off')
    plt.imshow(normal)
plt.show()
    
rows,cols = 1,4
plt.figure(figsize=(6,8))
for i_imgs in range(1, rows*cols+1):
    plt.subplot(rows,cols,i_imgs)
    pneumonia_path = opacity_train_file[i_imgs]
    pneumonia = cv2.imread(pneumonia_path)
    pneumonia = cv2.cvtColor(pneumonia,cv2.COLOR_BGR2RGB)  
    plt.title("Pneumonia")
    plt.axis('off')
    plt.imshow(pneumonia)
plt.show()

# prepare data (pandas frame)
IMG_SIZE = 256
IMG_CHANNELS = 3

df_train_n = pd.DataFrame(data={"filename": normal_train_file,
                                'labels' : '0'})
df_train_p = pd.DataFrame(data={"filename": opacity_train_file,
                                'labels' : '1'})
df_train = [df_train_n, df_train_p]
df_train = pd.concat(df_train)

df_test_n = pd.DataFrame(data={"filename": normal_test_file, 
                               'labels' : '0'})
df_test_p = pd.DataFrame(data={"filename": opacity_test_file, 
                               'labels' : '1'})
df_test = [df_test_n, df_test_p]
df_test = pd.concat(df_test)

df_val_n = pd.DataFrame(data={"filename": normal_val_file, 
                              'labels' : '0'})
df_val_p = pd.DataFrame(data={"filename": opacity_val_file, 
                              'labels' : '1'})
df_val = [df_val_n, df_val_p]
df_val = pd.concat(df_val)

print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)

# Data generation / Image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ref:
# https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

BATCH_SIZE = 16
data_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=False,
                            fill_mode='nearest')

datagen = ImageDataGenerator(rescale=1/255, **data_generator_args)

train_gen = datagen.flow_from_dataframe(dataframe=df_train,
                                        x_col="filename",
                                        y_col="labels",
                                        batch_size=BATCH_SIZE,
                                        seed=42,
                                        shuffle=True,
                                        class_mode="categorical",
                                        target_size=(IMG_SIZE,IMG_SIZE))
    
val_gen = datagen.flow_from_dataframe(dataframe=df_val,
                                      x_col="filename",
                                      y_col="labels",
                                      batch_size=BATCH_SIZE,
                                      seed=42,
                                      shuffle=True,
                                      class_mode="categorical",
                                      target_size=(IMG_SIZE,IMG_SIZE))

test_gen = datagen.flow_from_dataframe(dataframe=df_test,
                                       x_col="filename",
                                       y_col="labels",
                                       batch_size=BATCH_SIZE,
                                       seed=42,
                                       shuffle=True,
                                       class_mode="categorical",
                                       target_size=(IMG_SIZE,IMG_SIZE))

# Setting a model.
# The model used in this case is vgg16 pre-trained model
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = (IMG_SIZE,
                                                                        IMG_SIZE,
                                                                        IMG_CHANNELS))
# Define a model
def get_model(model):
    
    #for layer in vgg.layers[:-8]:  #Set block4 and block5 to be trainable.
    ##
    ## block 1-4 are sat as untrainable layers 
    ## they are features extraction layers
    for layer in model.layers[:-5]:
        # print(layer.name)
        layer.trainable = False #All others as non-trainable.

    x = model.output
    x = GlobalAveragePooling2D()(x) # GlobalAveragePooling and "NOT flatten". 
    x = Dense(2, activation="softmax")(x)  # Multiclass problem. 

    modified_model = Model(model.input, x)
    modified_model.compile(loss = "categorical_crossentropy", 
                           optimizer = Adam(lr=0.0001),
                           metrics=["accuracy"])
    return modified_model
model = get_model(vgg)
print(model.summary())

# Train the model
import tensorflow as tf
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_lung_pneumonia_localisation_vgg16.h5', verbose=1, save_best_only=True)
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

# Load the model
from tensorflow import keras
model = keras.models.load_model('model_lung_pneumonia_localisation_vgg16.h5')

from matplotlib.patches import Rectangle #To add a rectangle overlay to the image
from skimage.feature.peak import peak_local_max  #To detect hotspots in 2D images. 

# heatmap on X-ray to show the location with abnormality
def heatmap(img, model):
  
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred)
    #Get weights for all classes from the prediction layer
    last_layer_weights = model.layers[-1].get_weights()[0] #Prediction layer
    #Get weights for the predicted class.
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    #Get output from the last conv. layer
    last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
    last_conv_output = np.squeeze(last_conv_output)
    
    #Upsample/resize the last conv. output to same size as original image
    h = int(img.shape[0]/last_conv_output.shape[0])
    w = int(img.shape[1]/last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0]*img.shape[1], 512)), 
                 last_layer_weights_for_pred).reshape(img.shape[0],img.shape[1])
    
    #Since we have a lot of dark pixels where the edges may be thought of as 
    #high anomaly, let us drop all heat map values in this region to 0.
    #This is an optional step based on the image. 
    heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0
    
    #Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    #with rel threshold of 0.5 (compared to the max peak). 
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.3, min_distance=10) 
    
    return heat_map, peak_coords

# Show results
import random

rows,cols = 4,4
normal_list = random.sample(range(0, len(normal_test_file)), rows*cols)
pneumonia_list = random.sample(range(0, len(opacity_test_file)), rows*cols)

plt.figure(figsize=(12,12))
for i, i_img in enumerate(normal_list):
    plt.subplot(rows,cols,i+1)
    img_path = normal_test_file[i_img]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_AREA)
    heat_map, peak_coords = heatmap(img, model)
    pre_test = model.predict(np.expand_dims(img, axis=0))
    pre_test = np.argmax(pre_test, axis=1)
    if pre_test == 0: 
        pre_test='Normal' 
    else: 
        pre_test='Pneumonia'
    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3), cmap="gray")
    plt.imshow(heat_map, cmap='jet', alpha=0.3)
    plt.axis('off')
    title = "Normal, Predict: {}".format(pre_test)
    tit_obj = plt.title(title)
    if pre_test=='Pneumonia':
        plt.setp(tit_obj, color='r')
    for i in range(0,peak_coords.shape[0]):
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-25, y-25), 35,35,linewidth=1,edgecolor='r',facecolor='none'))
plt.show()

plt.figure(figsize=(15,15))
for i, i_img in enumerate(pneumonia_list):
    plt.subplot(rows,cols,i+1)
    img_path = opacity_test_file[i_img]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_AREA)
    heat_map, peak_coords = heatmap(img, model)
    pre_test = model.predict(np.expand_dims(img, axis=0))
    pre_test = np.argmax(pre_test, axis=1)
    if pre_test == 0: 
        pre_test='Normal' 
    else: 
        pre_test='Pneumonia'
    plt.imshow(img.astype('float32').reshape(img.shape[0],img.shape[1],3), cmap="gray")
    plt.imshow(heat_map, cmap='jet', alpha=0.3)
    plt.axis('off')
    title = "Pneumonia, Predict: {}".format(pre_test)
    tit_obj = plt.title(title)
    if pre_test=='Normal':
        plt.setp(tit_obj, color='r')
    for i in range(0,peak_coords.shape[0]):
        y = peak_coords[i,0]
        x = peak_coords[i,1]
        plt.gca().add_patch(Rectangle((x-25, y-25), 35,35,linewidth=1,edgecolor='r',facecolor='none'))
plt.show()
