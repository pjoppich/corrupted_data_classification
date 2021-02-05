import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
from scipy.special import binom
import warnings
try:
    import nifty6 as ift
except:
    warnings.warn("Failed importing nifty6")
from PIL import Image

def clear_axis():
  ax = plt.gca()
  ax.axes.yaxis.set_ticks([])
  ax.axes.xaxis.set_ticks([])

def convolution(colatitude):
    angle = colatitude * (180 / np.pi)
    return angle

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
n= 14

x_values = np.linspace(0, 1, n)
kernel = np.ones(n)
kernel = gaussian(x_values, 1, 3)
kernels = np.zeros(784)
for i in range(784//n):
  kernels[i*n:(i+1)*n] = kernel


def conv(colatitude):
    #plt.imshow(np.reshape(colatitude, [28, 28]))
    #GT = convolve(GT, kernel=[0, 0.5, 1, 2, 3.5, 5, 3.5, 2, 1, 0.5, 0], boundary='extend')
    return convolve(colatitude, kernel=[0.1, 0.5, 1, 2, 3.5, 5, 3.5, 2, 1, 0.5, 0.1], boundary='extend')


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def info_text(overlapping_nn, overlapping_dm):
  text = []
  text.append('----------------------------------------------------')
  text.append('{:<40} {}'.format('Key','Label'))
  for k, v in overlapping_nn.items():
      text.append("{:<40} {}".format(k, v))
  text.append('----------------------------------------------------')
  text.append('{:<40} {}'.format('Key','Label'))
  for k, v in overlapping_dm.items():
      text.append("{:<40} {}".format(k, v))
  return text

def get_noise(noise_level, position_space, seed):
    N_ift = ift.ScalingOperator(position_space, noise_level)
    with ift.random.Context(seed):
        n = N_ift.draw_sample_with_dtype(dtype=np.float64)
    return N_ift, n  # N respresents the noise operator (diagnonal covariance), n represents acutal sampled noise values
  
def rotation(image, img_shape, angle):
    im = np.reshape(image.val, img_shape)
    im = Image.fromarray(np.uint8(im*255))
    im = im.rotate(angle)
    im = np.asarray(im)/255
    im = np.reshape(im, image.shape)
    return ift.Field.from_raw(image.domain, im)
    
def split_validation_set(XTrain, YTrain, val_perc):
    '''
    Permutation of Training Dataset is taken from code in 
    an article pusblished on Medium: 
    https://medium.com/@mjbhobe/mnist-digits-classification-with-keras-ed6c2374bd0e
    Author: BhobeÃƒÆ’Ã‚Â©, Manish
    Date of Publication: 29.09.2018
    Relevant Code Section: Permutation of Data and Cut-Out of Validation Set
    Visit: 23.10.2020
    Minor modifications were made on val_percent and names of variables (adjusted to 
    my given variable names) and dimensionality of Datasets (mine is reshaped to vectors, 
    the author used 2D Arrays.)
    '''
    # shuffle the training dataset (5 times!)
    for i in range(5):
        np.random.seed(i)
        indexes = np.random.permutation(len(XTrain))

    XTrain = XTrain[indexes]
    YTrain = YTrain[indexes]

    # now set-aside 20% of the train_data/labels as the
    # cross-validation sets
    val_perc = 0.2
    val_count = int(val_perc * len(XTrain))

    # first pick validation set from train_data/labels
    XVal = XTrain[:val_count]
    YVal = YTrain[:val_count]

    # leave rest in training set
    XTrain = XTrain[val_count:]
    YTrain = YTrain[val_count:]
    
    return XTrain, YTrain, XVal, YVal
    
def dropout_uncertainty(model, data):
    data = torch.Tensor(np.reshape(data, [1, 1, 28, 28]))
    model.train()
    T = 50
    output_list = []
    with torch.no_grad():
                for i in range(T):
                    #output_list.append(torch.unsqueeze(F.softmax(model(data), dim=1), dim=0))
                    output_list.append(torch.unsqueeze(model(data), dim=0))
                output_mean = torch.cat(output_list, 0).mean(dim=0)
                output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
                output_variance = torch.cat(output_list, 0).var(dim=0)
                confidence = output_mean.data.cpu().numpy().max()
                predict = output_mean.data.cpu().numpy().argmax()
                predict = output_mean
                return predict, output_variance


