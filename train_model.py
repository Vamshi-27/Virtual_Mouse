# Importing required modules
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from skimage import io

# Check directories
print(os.listdir("data/"))

# Set train directory
train_dir = "data/"

# Output size = number of gesture classes
outputSize = len(os.listdir(train_dir)) 
epochs = 30

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training data loader
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

# Function to create model
def create_model(outputSize):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    
    model.add(GlobalAveragePooling2D())  # replaces Flatten()
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(outputSize, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build model
model = create_model(outputSize)
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs
)

# Save model and weights
model.save_weights('gesture_model.weights.h5')
model.save("gesture_model.h5")
