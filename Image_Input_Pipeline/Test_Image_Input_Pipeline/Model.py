import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

import tensorflow as tf
from LoadImageData import LoadImages
from ConfigureDataForFinerControl import CreateInputPipeline
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
loaded_ds = LoadImages(32, 180, 180, url, 'flower_photos')
loaded_ds, image_count = loaded_ds.create_direction()

input_object = CreateInputPipeline(loaded_ds, 0.8, image_count, 1000, 32, 180, 180)
train_ds, test_ds = input_object.execute()

num_classes = 5

model = Sequential([
  layers.Rescaling(1./255, input_shape=(180, 180, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=2
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



