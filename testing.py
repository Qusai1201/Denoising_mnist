import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from sklearn.metrics import log_loss
from processing import *


( _ , _ ), (test_data, _) = mnist.load_data()


test_data = preprocess(test_data)

noisy_test_data = noise(test_data , 0.5)



autoencoder = tf.keras.models.load_model('autoencoder.h5')


test_loss = autoencoder.evaluate(test_data , noisy_test_data)
print(test_loss)

predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions , 10) 


