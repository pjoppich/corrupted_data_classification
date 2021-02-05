'''
    The Code for creating a 'convolution' [Conv.py] is mainly based on the following
    GitHub Repository "Convolution as Matrix Multiplication":
    https://github.com/alisaaalehi/convolution_as_multiplication
    Author: Salehi, Ali, [https://github.com/alisaaalehi]
    Date of last commit by author: 08. Jun 2019
    Visited: 21.09.2020
    
    Modifications were only made on output shape and used filter/convolution matrix. 
'''
import numpy as np
import scipy
from scipy.linalg import toeplitz
import nifty6 as ift


def sobel(amplitude, position_space):
    F = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]]) * amplitude # sobel
    
    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    try:
        output_row_num = position_space.shape[0] 
        output_col_num = position_space.shape[1]
    except:
        output_row_num = np.int(np.sqrt(position_space.shape[0]))
        output_col_num = output_row_num

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                           (0, output_col_num - F_col_num)),
                        'constant', constant_values=0)

    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
      c = F_zero_padded[i, :] # i th row of the F 
      r = np.r_[c[0], np.zeros(output_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
      toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
      toeplitz_list.append(toeplitz_m)


    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(output_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
          start_i = i * b_h
          start_j = j * b_w
          end_i = start_i + b_h
          end_j = start_j + b_w
          doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
    
    conv_matrix = doubly_blocked
    if len(position_space.shape)==3:

        padded_conv = np.zeros([3072, 3072])
        padded_conv[:1024, :1024] = conv_matrix
        padded_conv[1024:2048, 1024:2048] = np.eye(1024)
        padded_conv[2048:, 2048:] = np.eye(1024)

        padded_conv1 = np.eye(3072)
        for i in range(0, 3072, 3):
            for j in range(0, 3072, 3):
                padded_conv1[i, j] = conv_matrix[i//3, j//3]

        padded_conv2 = np.eye(3072)
        for i in range(1, 3072, 3):
            for j in range(1, 3072, 3):
                padded_conv2[i, j] = conv_matrix[i//3, j//3]

        padded_conv3 = np.eye(3072)
        for i in range(2, 3072, 3):
            for j in range(2, 3072, 3):
                padded_conv3[i, j] = conv_matrix[i//3, j//3]

        C1 = ift.MatrixProductOperator(position_space, padded_conv1, flatten=True)
        C2 = ift.MatrixProductOperator(position_space, padded_conv2, flatten=True)
        C3 = ift.MatrixProductOperator(position_space, padded_conv3, flatten=True)

        C = C1@C2@C3
    
        return C
    
    else:
        return ift.MatrixProductOperator(position_space, conv_matrix)



def gaussian_blur(kernel_size, amplitude, position_space):
    def gkern(l=5, sig=1.):
        """\
        
        Copyright: https://stackoverflow.com/a/43346070
        21.11.2020
        creates gaussian kernel with side length l and a sigma of sig
        """

        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        return kernel / np.sum(kernel)
    F = gkern(l=kernel_size, sig=amplitude)
    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    try:
        output_row_num = position_space.shape[0] 
        output_col_num = position_space.shape[1]
    except:
        output_row_num = np.int(np.sqrt(position_space.shape[0]))
        output_col_num = output_row_num

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                           (0, output_col_num - F_col_num)),
                        'constant', constant_values=0)

    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
      c = F_zero_padded[i, :] # i th row of the F 
      r = np.r_[c[0], np.zeros(output_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
      toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
      toeplitz_list.append(toeplitz_m)


    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(output_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
          start_i = i * b_h
          start_j = j * b_w
          end_i = start_i + b_h
          end_j = start_j + b_w
          doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
        conv_matrix = doubly_blocked
    if len(position_space.shape)==3:

        padded_conv = np.zeros([3072, 3072])
        padded_conv[:1024, :1024] = conv_matrix
        padded_conv[1024:2048, 1024:2048] = np.eye(1024)
        padded_conv[2048:, 2048:] = np.eye(1024)

        padded_conv1 = np.eye(3072)
        for i in range(0, 3072, 3):
            for j in range(0, 3072, 3):
                padded_conv1[i, j] = conv_matrix[i//3, j//3]

        padded_conv2 = np.eye(3072)
        for i in range(1, 3072, 3):
            for j in range(1, 3072, 3):
                padded_conv2[i, j] = conv_matrix[i//3, j//3]

        padded_conv3 = np.eye(3072)
        for i in range(2, 3072, 3):
            for j in range(2, 3072, 3):
                padded_conv3[i, j] = conv_matrix[i//3, j//3]

        C1 = ift.MatrixProductOperator(position_space, padded_conv1, flatten=True)
        C2 = ift.MatrixProductOperator(position_space, padded_conv2, flatten=True)
        C3 = ift.MatrixProductOperator(position_space, padded_conv3, flatten=True)

        C = C1@C2@C3
    
        return C
    
    else:
        return ift.MatrixProductOperator(position_space, conv_matrix)

def edge_detection(amplitude, position_space):
    F = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])*1/16 * amplitude # Edge-Detection
       # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    try:
        output_row_num = position_space.shape[0] 
        output_col_num = position_space.shape[1]
    except:
        output_row_num = np.int(np.sqrt(position_space.shape[0]))
        output_col_num = output_row_num

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                           (0, output_col_num - F_col_num)),
                        'constant', constant_values=0)

    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
      c = F_zero_padded[i, :] # i th row of the F 
      r = np.r_[c[0], np.zeros(output_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
      toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
      toeplitz_list.append(toeplitz_m)


    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(output_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
          start_i = i * b_h
          start_j = j * b_w
          end_i = start_i + b_h
          end_j = start_j + b_w
          doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
        conv_matrix = doubly_blocked
    if len(position_space.shape)==3:

        padded_conv = np.zeros([3072, 3072])
        padded_conv[:1024, :1024] = conv_matrix
        padded_conv[1024:2048, 1024:2048] = np.eye(1024)
        padded_conv[2048:, 2048:] = np.eye(1024)

        padded_conv1 = np.eye(3072)
        for i in range(0, 3072, 3):
            for j in range(0, 3072, 3):
                padded_conv1[i, j] = conv_matrix[i//3, j//3]

        padded_conv2 = np.eye(3072)
        for i in range(1, 3072, 3):
            for j in range(1, 3072, 3):
                padded_conv2[i, j] = conv_matrix[i//3, j//3]

        padded_conv3 = np.eye(3072)
        for i in range(2, 3072, 3):
            for j in range(2, 3072, 3):
                padded_conv3[i, j] = conv_matrix[i//3, j//3]

        C1 = ift.MatrixProductOperator(position_space, padded_conv1, flatten=True)
        C2 = ift.MatrixProductOperator(position_space, padded_conv2, flatten=True)
        C3 = ift.MatrixProductOperator(position_space, padded_conv3, flatten=True)

        C = C1@C2@C3
    
        return C
    
    else:
        return ift.MatrixProductOperator(position_space, conv_matrix)
    
def own(amplitude, conv_matrix, position_space):
    F = conv_matrix*amplitude
        # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    try:
        output_row_num = position_space.shape[0] 
        output_col_num = position_space.shape[1]
    except:
        output_row_num = np.int(np.sqrt(position_space.shape[0]))
        output_col_num = output_row_num

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                           (0, output_col_num - F_col_num)),
                        'constant', constant_values=0)

    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
      c = F_zero_padded[i, :] # i th row of the F 
      r = np.r_[c[0], np.zeros(output_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                        # the result is wrong
      toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
      toeplitz_list.append(toeplitz_m)


    # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(output_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
          start_i = i * b_h
          start_j = j * b_w
          end_i = start_i + b_h
          end_j = start_j + b_w
          doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
        conv_matrix = doubly_blocked
    if len(position_space.shape)==3:

        padded_conv = np.zeros([3072, 3072])
        padded_conv[:1024, :1024] = conv_matrix
        padded_conv[1024:2048, 1024:2048] = np.eye(1024)
        padded_conv[2048:, 2048:] = np.eye(1024)

        padded_conv1 = np.eye(3072)
        for i in range(0, 3072, 3):
            for j in range(0, 3072, 3):
                padded_conv1[i, j] = conv_matrix[i//3, j//3]

        padded_conv2 = np.eye(3072)
        for i in range(1, 3072, 3):
            for j in range(1, 3072, 3):
                padded_conv2[i, j] = conv_matrix[i//3, j//3]

        padded_conv3 = np.eye(3072)
        for i in range(2, 3072, 3):
            for j in range(2, 3072, 3):
                padded_conv3[i, j] = conv_matrix[i//3, j//3]

        C1 = ift.MatrixProductOperator(position_space, padded_conv1, flatten=True)
        C2 = ift.MatrixProductOperator(position_space, padded_conv2, flatten=True)
        C3 = ift.MatrixProductOperator(position_space, padded_conv3, flatten=True)

        C = C1@C2@C3
    
        return C
    
    else:
        return ift.MatrixProductOperator(position_space, conv_matrix)