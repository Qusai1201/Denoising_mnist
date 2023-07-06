import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from processing import *

(train_data, _), (test_data, _) = mnist.load_data()

train_data = preprocess(train_data)
test_data = preprocess(test_data)

noisy_train_data = noise(train_data , 0.5)
noisy_test_data = noise(test_data , 0.5)

display(train_data, noisy_train_data)



autoencoder = tf.keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),

        # Encoder
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2), padding="same"),

        # Decoder
        layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),\
        
        layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
    ]
)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
   filepath='autoencoder.h5',
   monitor='val_loss',
   mode='min',
   save_best_only=True
)

history = autoencoder.fit(
   x=noisy_train_data,
   y=train_data,
   epochs=50,
   batch_size=128,
   shuffle=True,
   validation_data=(noisy_test_data, test_data),
   callbacks=[model_checkpoint_callback]
)

fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.xticks(ticks=np.arange(len(history.history['loss'])), labels=np.arange(1, len(history.history['loss'])+1))
plt.legend()

plt.show()

