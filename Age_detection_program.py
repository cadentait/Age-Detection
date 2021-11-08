import shutil, os
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

#Extract labels of class names from .csv file.
data_csv = pd.read_csv('/Users/cadentait/python/github portfolio/Age Detection/train.csv')
labels = data_csv.sort_values('Class')
class_names = list(labels.Class.unique())

#Paths to training images.
train_images = pathlib.Path('/Users/cadentait/python/age-detection/Age-Detection/Train_ copy')
data_dir = pathlib.Path('/Users/cadentait/python/age-detection/Age-Detection/Train_')

#Standardize images from dataset.
batch_size = 32 
img_height = 180
img_width = 180

#Training dataset.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split =0.2,
    subset='training',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
    )

#Validation dataset.
train_val = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size
)

#Set model variables.
class_names = train_ds.class_names
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)

#Normalize the dataset.
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_bath = next(iter(normalized_ds))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
train_val = train_val.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 3

#Build model.
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), padding = 'same'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), padding = 'same'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2), padding = 'same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

#Compile model
model.compile(
    optimizer = 'adam',
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = 'accuracy'
)

model.fit(
    train_ds,
    validation_data = train_val,
    epochs = 3,
)

#Print summary of the model.
model.summary()