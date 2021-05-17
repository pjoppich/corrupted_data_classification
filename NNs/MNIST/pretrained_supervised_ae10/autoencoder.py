# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sklearn as sk
from sklearn import decomposition
import numpy as np

# Colab and system related
import os
import time

import sys
sys.path.append('./corrupted_data_classification/helper_functions/')
from helper_functions import split_validation_set
#GPU

import tensorflow as tf

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

# Commented out IPython magic to ensure Python compatibility.
def plot_pca(Encoder, input_data):
    # Get latent space representation (predict samples up to 'bottleneck')
    hidden_rep = Encoder.predict(input_data)

    # For visualization, compute PCA of latent space
    # pca = sk.decomposition.KernelPCA(n_components=10, kernel='linear')
    pca = sk.decomposition.PCA(n_components=10)
    principalComponents = pca.fit_transform(hidden_rep)
    labels = np.arange(0, 10, dtype=int)

    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    digits = range(0, 10, 1)

    # Plot 3 strongest PCs
    for digit in digits:
        PC1 = principalComponents[np.where(YVal == digit), 0]
        PC2 = principalComponents[np.where(YVal == digit), 1]
        #PC3 = principalComponents[np.where(YTest == digit), 2]

        plot_indices = random.sample(range(0, len(PC1.T)), round(len(PC1.T) / 1))
        ax.scatter(PC1[:, plot_indices],
                   PC2[:, plot_indices], marker='.', s=3)
                   #PC3[:, plot_indices], label=digit)
    plt.legend(labels)

# Plotting
import matplotlib as mpl

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
# %matplotlib inline

#rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#!apt install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng
mpl.rcParams['figure.dpi']= 1000
mpl.rcParams['font.size'] = 9.0
plot_pca(Encoder, XVal)

plt.show()

def plot_reconstruction(Encoder, model, input_data, input_data_labels):
    # Plotting reconstructions through autoencoder
    predicted_img = model.predict(input_data)
    predicted_img = np.array(predicted_img[0][:][:])
    print(len(input_data))
    random_image_indices = random.sample(range(len(input_data)), 18)

    fig, ax = plt.subplots(18, 2)
    for i in range(1, 19):
        if i % 2 != 0:
            plt.subplot(9, 2, i)
            plt.imshow(np.reshape(input_data[random_image_indices[i], :], [28, 28]), cmap='gray')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_ticks([])
            plt.ylabel('Labeled: {}'.format(input_data_labels[random_image_indices[i]]), fontsize=10, rotation=0,
                       labelpad=35)

        if i % 2 == 0:
            plt.subplot(9, 2, i)
            img = predicted_img[random_image_indices[i - 1]].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            img = img.reshape(784).tolist()
            img_class = Encoder.predict([img[:]])
            img_class = np.array(img_class[0][:])
            img_class = np.where(img_class == np.amax(img_class))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_ticks([])
            plt.ylabel('Classified: {}'.format(img_class[0][0]), fontsize=10, rotation=0, labelpad=35)
        if i == 1:
            plt.title('Original Images')
        if i == 2:
            plt.title('Generated Images')
    plt.tight_layout(pad=0.1)


plot_reconstruction(Encoder, model, XTest, YTest)

plt.show()

