# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# Commented out IPython magic to ensure Python compatibility.
# Colab and system related
import os
import sys
###
# Necessary to convert tensorflow-object (e.g. Neural Network) to Nifty-Operator
sys.path.append('corrupted_data_classification/helper_functions/')

###
import tensorflow as tf
# Include path to access helper functions and Mask / Conv Operator
sys.path.append('corrupted_data_classification/helper_functions/')
from helper_functions import clear_axis, gaussian, get_cmap, info_text, get_noise, rotation, split_validation_set
import Mask # Masking Operator
import Conv # Convolution Operator
sys.path.remove
# Tensorflow

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower

# Numerics
import random
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import sklearn as sk
from sklearn import decomposition

# Load MNIST Dataset
mnist = tf.keras.datasets.mnist
(XTrain, YTrain), (XTest, YTest) = mnist.load_data()
XTrain, XTest = XTrain / 255.0, XTest / 255.0
# Cut out last 100 Training images for comparison
XTrain = XTrain[0:-100]
YTrain = YTrain[0:-100]

# Reshape Xtrain and XTest to 1x784 Vectors instead of 28x28 arrays
XTrain = XTrain.reshape((len(XTrain), np.prod(XTrain.shape[1:])))
XTest = XTest.reshape((len(XTest), np.prod(XTest.shape[1:])))

XTrain, YTrain, XVal, YVal = split_validation_set(XTrain, YTrain, val_perc=0.2)

def autoencoder_deep(latent_space_size):
    Input = tf.keras.layers.Input(shape=784)
    h1 = tf.keras.layers.Dense(512, activation='selu', kernel_initializer='lecun_normal')(Input)
    h2 = tf.keras.layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(h1)
    h3 = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal')(h2)
    encoded = tf.keras.layers.Dense(latent_space_size, activation='linear', 
    activity_regularizer=tf.keras.regularizers.L2(0.001))(h3)
    # Decoder
    Decoder_Input = tf.keras.layers.Input(shape=latent_space_size)  # Input for Decoder
    h5 = tf.keras.layers.Dense(128, activation='selu', kernel_initializer='lecun_normal')(Decoder_Input)
    h6 = tf.keras.layers.Dense(256, activation='selu', kernel_initializer='lecun_normal')(h5)
    h7 = tf.keras.layers.Dense(512, activation='selu', kernel_initializer='lecun_normal')(h6)
    decoded = tf.keras.layers.Dense(784, activation='sigmoid')(h7)

    # Decouple Encoder and Decoder from overall model
    Encoder = tf.keras.Model(Input, encoded)
    Decoder = tf.keras.Model(Decoder_Input, decoded)
    decoded = Decoder(encoded)
    model = tf.keras.Model(Input, [decoded, encoded])
    return Encoder, Decoder, model


Encoder, Decoder, model = autoencoder_deep(10)

# Loss Function for Reconstruction of images (i.e. overall Autoencoder)
def loss_fn_AE(y_true, y_pred):
    # y_pred = tf.nn.elu(y_pred) * tf.nn.softplus(y_pred)
    # return tf.losses.categorical_crossentropy(y_true, y_pred)
    # y_pred = tf.nn.softmax(y_pred)
    return  tf.losses.binary_crossentropy(y_true,y_pred)
    #return  tf.keras.losses.MeanSquaredError(y_true, y_pred)
# Loss Function for Classification of Images in latent space
def loss_fn_Encoder(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    return tf.losses.sparse_categorical_crossentropy(y_true, y_pred)

# Training Options
model.compile(optimizer='adam',
              #loss=[loss_fn_AE, loss_fn_Encoder],
              loss=[loss_fn_AE, loss_fn_Encoder], 
              metrics=['accuracy'])

# Training and Testing
results = model.fit(XTrain, [XTrain, YTrain], epochs=25)
model.evaluate(XTest, [XTest, YTest], verbose=2)

# Save trained Decoder and trained Encoder
Decoder.save('./corrupted_data_classification/NNs/MNIST/pretrained_supervised_ae10/Decoder/', save_format='tf')
Encoder.save('./corrupted_data_classification/NNs/MNIST/pretrained_supervised_ae10/Encoder/', save_format='tf')

plt.plot(results.history['dense_3_accuracy'])

