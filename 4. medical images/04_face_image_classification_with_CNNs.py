#!/usr/bin/env python
# coding: utf-8

# # Face Image Classification using CNN models
# ## 1. Data is in .csv file
# ### Use pandas library to process this data
# ## 2. Build CNN models
# ### CNN models are used to process image data.
# ## 3. Data visulisation
# ### Display face images and subjects' age, gender and ethnicity.
# ## 4. Do data analysis
# ### Such as mean values.
# 
# ### Warning: data didn't provide the names of labels in gender and ethnicity.
# ### URL: https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

# In[1]:


# Load relevant library to process data
import pandas as pd

# Load data from csv file as pandas data frame.
df_data = pd.read_csv('data/age_gender.csv')


# # Data Preprocessing:

# In[2]:


# show data frame:
df_data


# In[3]:


# img_name is useless in this practice
# This column will be removed
# Use "drop" function
df_data = df_data.drop('img_name', axis=1)
df_data


# In[4]:


# Check Null/Nan data in the document
print(df_data.isnull().sum())

# if there are any Nan in the document
# use:
# df_data.dropna()
# df_data

# Show simple analysis including count, mean, std, min and max.
print(df_data.describe())


# In[5]:


# Plot data as image for visualisation
# using pandas library
import matplotlib.pyplot as plt

plt.figure(figsize=(6,12))
df_data['age'].plot(kind='hist', 
                    bins=30,
                    legend=True,
                    title='Age Counting',
                    ylabel='Counts',
                    xlabel='Ages',
                    figsize=(6,5))
plt.show()

plt.figure(figsize=(6,12))
df_data['age'].plot(kind='box',
                    legend=True,
                    title='Age distribution',
                    ylabel='Distribution',
                    xlabel='',
                    figsize=(6,5))
plt.show()


# # Data preparation
# ### prepare data for training CNN models to classify face images

# In[6]:


# Divided 'age' into four ranges
df_data['age'] = pd.qcut(df_data['age'],
                         q=4,
                         labels=[0, 1, 2, 3])

df_data


# In[7]:


import numpy as np

# Check the size of face images
# they are in 'pixels' column
num_pixels = len(df_data['pixels'][0].split(' '))
IMG_SIZE = int(np.sqrt(num_pixels))
print('SIZE of image: ', IMG_SIZE)


# In[8]:


target_columns = ['gender', 'ethnicity', 'age']

y = df_data[target_columns]
X = df_data.drop(target_columns, axis=1)


# In[9]:


y


# In[10]:


X


# In[11]:


# Prepare face images data for train, test and validation

img = []
for i in range(len(df_data)):
    x_img = df_data['pixels'][i].split(' ')
    x_img = np.array(x_img)
    x_img = np.reshape(x_img, (IMG_SIZE, IMG_SIZE))
    img.append(x_img)
img = np.array(img).astype('float64')/255
img = np.expand_dims(img, axis=3)

print("data size: ",img.shape)


# # Image visualisation

# In[12]:


import random

plt.figure(figsize=(15, 15))
rows,cols = 4,4
lt = random.sample(range(0, img.shape[0]), rows*cols)

for index, image in enumerate(lt):
    plt.subplot(rows, cols, index + 1)
    plt.imshow(img[image],cmap='gray')
    plt.axis('off')
    plt.title(
        "Age range:"+str(y['age'].iloc[image])+
        "  Ethnicity:"+str(y['ethnicity'].iloc[image])+
        "  Gender:"+ str(y['gender'].iloc[image])
    )

plt.show()


# In[13]:

# prepare labels
y_age = pd.get_dummies(y['age'])
y_eth = pd.get_dummies(y['ethnicity'])
y_gender = pd.get_dummies(y['gender'])

print(y_age.shape)
print(y_eth.shape)
print(y_gender.shape)


# # Model preparation (CNN)
# In[16]:
## Define model

import tensorflow as tf
def build_model(num_classes, activation='softmax', loss='sparse_categorical_crossentropy'):
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

# In[17]:
model_age = build_model(4, activation='softmax', loss='categorical_crossentropy')
model_eth = build_model(5, activation='softmax', loss='categorical_crossentropy')
model_gnd = build_model(2, activation='softmax', loss='categorical_crossentropy')

print(model_age.summary())
print(model_eth.summary())
print(model_gnd.summary())
# In[18]:

checkpointer_age = tf.keras.callbacks.ModelCheckpoint('vgg19_age.h5', verbose=1, save_best_only=True)
checkpointer_eth = tf.keras.callbacks.ModelCheckpoint('vgg19_eth.h5', verbose=1, save_best_only=True)
checkpointer_gnd = tf.keras.callbacks.ModelCheckpoint('vgg19_gnd.h5', verbose=1, save_best_only=True)

callbacks_age = [checkpointer_age,
                 tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
callbacks_eth = [checkpointer_eth,
                 tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]
callbacks_gnd = [checkpointer_gnd,
                 tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')]

# # Set tp train, test data
# In[19]:
from sklearn.model_selection import train_test_split
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(img, 
                                                                    y_age, 
                                                                    train_size=0.7,
                                                                    shuffle=True)
X_eth_train, X_eth_test, y_eth_train, y_eth_test = train_test_split(img, 
                                                                    y_eth, 
                                                                    train_size=0.7,
                                                                    shuffle=True)
X_gnd_train, X_gnd_test, y_gnd_train, y_gnd_test = train_test_split(img, 
                                                                    y_gender, 
                                                                    train_size=0.7,
                                                                    shuffle=True)

# # Start training
# In[20]:
history_age = model_age.fit(X_age_train,
                            y_age_train,
                            validation_split=0.2,
                            batch_size=64,
                            epochs=100,
                            callbacks=callbacks_age,
                            verbose=1)

history_eth = model_eth.fit(X_eth_train,
                            y_eth_train,
                            validation_split=0.2,
                            batch_size=64,
                            epochs=100,
                            callbacks=callbacks_age,
                            verbose=1)

history_gnd = model_gnd.fit(X_gnd_train,
                            y_gnd_train,
                            validation_split=0.2,
                            batch_size=64,
                            epochs=100,
                            callbacks=callbacks_age,
                            verbose=1)

# # Show results.
# In[20]:
plt.title('Learning Curves (Age)')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history_age.history['loss'], label='train')
plt.plot(history_age.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('Learning Curves (Eth)')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history_eth.history['loss'], label='train')
plt.plot(history_eth.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('Learning Curves (Gnd)')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history_gnd.history['loss'], label='train')
plt.plot(history_gnd.history['val_loss'], label='val')
plt.legend()
plt.show()

# In[21]:
acc_age = model_age.evaluate(X_age_test,
                             y_age_test)
acc_eth = model_eth.evaluate(X_eth_test,
                             y_eth_test)
acc_gnd = model_gnd.evaluate(X_gnd_test,
                             y_gnd_test)

print("Accuracy (Age): ", int(acc_age[1]*100), "%")
print("Accuracy (Ethnicity): ", int(acc_eth[1]*100), "%")
print("Accuracy (Gender): ", int(acc_gnd[1]*100), "%")

# # Prediction
# In[22]:
prediction_age = model_age.predict(X_age_test)
prediction_age = np.argmax(prediction_age,axis=1)
plt.figure(figsize=(15, 15))
rows,cols = 4,4
lt = random.sample(range(0, len(prediction_age)), rows*cols)
for index, image in enumerate(lt):
    plt.subplot(rows, cols, index + 1)
    plt.imshow(X_age_test[image],cmap='gray')
    plt.axis('off')
    y_age_test_label = np.argmax(np.array(y_age_test),axis=1)
    tit_obj = plt.title(
        "Age label:"+str(y_age_test_label[image])+
        "  Predict:"+str(prediction_age[image])
        )
    if y_age_test_label[image] != prediction_age[image]:
        plt.setp(tit_obj, color='r')
plt.show()


prediction_eth = model_eth.predict(X_eth_test)
prediction_eth = np.argmax(prediction_eth,axis=1)
plt.figure(figsize=(15, 15))
rows,cols = 4,4
lt = random.sample(range(0, len(prediction_eth)), rows*cols)
for index, image in enumerate(lt):
    plt.subplot(rows, cols, index + 1)
    plt.imshow(X_eth_test[image],cmap='gray')
    plt.axis('off')
    y_eth_test_label = np.argmax(np.array(y_eth_test),axis=1)
    tit_obj = plt.title(
        "Ethnicity label:"+str(y_eth_test_label[image])+
        "  Predict:"+str(prediction_eth[image])
        )
    if y_eth_test_label[image] != prediction_eth[image]:
        plt.setp(tit_obj, color='r')
plt.show()


prediction_gnd = model_gnd.predict(X_gnd_test)
prediction_gnd = np.argmax(prediction_gnd,axis=1)
plt.figure(figsize=(15, 15))
rows,cols = 4,4
lt = random.sample(range(0, len(prediction_gnd)), rows*cols)
for index, image in enumerate(lt):
    plt.subplot(rows, cols, index + 1)
    plt.imshow(X_gnd_test[image],cmap='gray')
    plt.axis('off')
    y_gnd_test_label = np.argmax(np.array(y_gnd_test),axis=1)
    tit_obj = plt.title(
        "Gender label:"+str(y_gnd_test_label[image])+
        "  Predict:"+str(prediction_gnd[image])
        )
    if y_gnd_test_label[image] != prediction_gnd[image]:
        plt.setp(tit_obj, color='r')
plt.show()


