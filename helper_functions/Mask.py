import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
from scipy.special import binom
import nifty6 as ift

def no_mask(position_space):
    mask = np.ones(position_space.shape)
    mask = ift.Field.from_raw(position_space, mask)
    M = ift.DiagonalOperator(mask)
    return M



def checkerboard_mask(position_space, mask_range):
    x_shape = np.sqrt(position_space.shape)
    y_shape = np.sqrt(position_space.shape)
    xy_shape = position_space.shape
    checkerboard_x = np.tile(np.array([1, 1, 0, 0]), 196)
    checkerboard = np.reshape(checkerboard_x, [x_shape, y_shape]) * np.reshape(checkerboard_x, [x_shape, y_shape]).T
    mask = np.reshape(checkerboard, xy_shape)
    mask = ift.Field.from_raw(position_space, mask)
    M = ift.DiagonalOperator(mask)

    return M


def half_mask(position_space, mask_range):
    mask = np.ones(position_space.shape)
    x_shape = np.sqrt(position_space.shape)[0]
    xy_shape = position_space.shape[0]
    try:
        z_shape = position_space.shape[2]
    except:
        z_shape = 0
    Flag = False
    for i in range(xy_shape):
        if ((i - np.round(mask_range * x_shape)) % x_shape) == 0 or (i % x_shape) == 0:
            Flag = not Flag
        if Flag == False:
            mask[i] = 0
        else:
            mask[i] = 1
    if z_shape != 0:
        xy_shape = position_space.shape[0] * position_space.shape[1]
        x_shape = position_space.shape[0]
        mask = np.ones([xy_shape, z_shape])
        for z in range(z_shape):
            Flag = True
            for i in range(xy_shape):
                print(i)
                if ((i - np.round(mask_range * x_shape)) % x_shape) == 0 or (i % x_shape) == 0:
                    Flag = not Flag
                if Flag == False:
                    mask[i, z] = 0
                else:
                    mask[i, z] = 1
    mask = np.reshape(mask, position_space.shape)
    mask = ift.Field.from_raw(position_space, mask)
    M = ift.DiagonalOperator(mask)
    return M



def corner_mask(position_space, mask_range):
    # Checkerboard mask for 2D mode
    x_shape = np.sqrt(position_space.shape)[0]
    y_shape = np.sqrt(position_space.shape)[0]
    xy_shape = position_space.shape[0]
    try:
        z_shape = position_space.shape[2]
    except:
        z_shape = 0
    Flag = False
    mask = np.ones(position_space.shape)
    for i in range(xy_shape):
        if ((i - np.round(mask_range * x_shape)) % x_shape) == 0 or (i % x_shape) == 0:
            Flag = not Flag
        if Flag == False or (i >= xy_shape / 2):
            mask[i] = 0
        else:
            mask[i] = 1
    if z_shape != 0:
        xy_shape = position_space.shape[0] * position_space.shape[1]
        x_shape = position_space.shape[0]
        mask = np.ones([xy_shape, z_shape])
        for z in range(z_shape):
            Flag = True
            for i in range(xy_shape):
                if ((i - np.round(mask_range * x_shape)) % x_shape) == 0 or (i % x_shape) == 0:
                    Flag = not Flag
                if Flag == False or (i >= xy_shape / 2):
                    mask[i, z] = 0
                else:
                    mask[i, z] = 1
    mask = np.reshape(mask, [position_space.shape[0], position_space.shape[1], position_space.shape[2]])
    mask = ift.Field.from_raw(position_space, mask)
    M = ift.DiagonalOperator(mask)
    return M
def window_mask(position_space, mask_range):
    mask = np.ones(position_space.shape)
    x_shape = np.sqrt(position_space.shape)[0]
    xy_shape = position_space.shape[0]
    try:
        z_shape = position_space.shape[2]
    except:
        z_shape = 0
    Flag = False
    for i in range(xy_shape):
        if i%x_shape==mask_range or i%x_shape==x_shape-mask_range:
            Flag = not Flag
        if Flag == False:
            mask[i] = 0
        else:
            mask[i] = 1
    if z_shape != 0:
        xy_shape = position_space.shape[0] * position_space.shape[1]
        x_shape = position_space.shape[0]
        mask = np.ones([xy_shape, z_shape])
        for z in range(z_shape):
            Flag = True
            for i in range(xy_shape):
                print(i)
                if ((i - np.round(mask_range * x_shape)) % x_shape) == 0 or (i % x_shape) == 0:
                    Flag = not Flag
                if Flag == False:
                    mask[i, z] = 0
                else:
                    mask[i, z] = 1
    mask[0:np.int(mask_range*x_shape)]=0
    mask[np.int(xy_shape)-(mask_range*np.int(x_shape)):]=0
    mask = np.reshape(mask, position_space.shape)
    mask = ift.Field.from_raw(position_space, mask)
    M = ift.DiagonalOperator(mask)
    return M
###
# [2]
###
def random_mask(n_blobs, seed, position_space):
    ''' 
    The Code for creating a 'random mask' is mainly based on the following
    StackOverflow Answer published under CreativeCommons 4.0:
    https://stackoverflow.com/a/50751932
    Author: ImportanceOfBeingErnest [https://stackoverflow.com/users/4124317/importanceofbeingernest]
    Date of Pubilshing: 08. Jun 2018
    Visited: 10.09.2020
    Several modifications were made on the originally published code. Among others, "blobs" are filled
    with color, dimensions are adjusted to this use-case. 
    '''
    
    # Plotting-Output is suppressed by plt.ioff(). Plotting is necessary for creating a random mask.
    plt.ioff()
    def get_curve(points, **kw):
        segments = []
        for i in range(len(points) - 1):
            seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def ccw_sort(p):
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]
    
    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

    def bezier(points, num=200):
      N = len(points)
      t = np.linspace(0, 1, num=num)
      curve = np.zeros((num, 2))
      for i in range(N):
          curve += np.outer(bernstein(N - 1, i, t), points[i])
      return curve

    class Segment():
      def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

      def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)
    def get_bezier_curve(a, rad=0.2, edgy=0):
        np.random.seed(10)
        """ given an array of points *a*, create a curve through
        those points. 
        *rad* is a number between 0 and 1 to steer the distance of
          control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
        p = np.arctan(edgy) / np.pi + .5
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=rad, method="var")
        x, y = c.T
        return x, y, a

    def get_random_points(n=5, scale=0.8, mindst=5, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7 / n
        a = np.random.rand(n, 2)
        d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
        if np.all(d >= mindst) or rec >= 200:
            return a * scale
        else:
            return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)

    fig = plt.figure()
    rad = 0.5
    edgy = 0.6
    random.seed(seed)

    for i, c in enumerate([[random.uniform(0, 1) for x in range(2)] for y in range(n_blobs)]):
        np.random.seed(i + seed)
        a = get_random_points(n=7, scale=0.2) + c
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)

        plt.plot(x, y, c='black')
        plt.fill_between(x, y)
        plt.axis('off')
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = (data[:, :, 0] + data[:, :, 1] + data[:, :, 2]) / 3
    data = data / np.max(data)
    data[data > 0.99] = 1
    data[data != 1] = 0

    data = resize(data, [50, 50])
    data[data < 0.75] = 0
    data[data >= 0.75] = 1
    if 50 - position_space.shape[0] - 10 > 0:
        data = data[50 - position_space.shape[0] - 10:50 - 10, 50 - position_space.shape[0] - 10:50 - 10]
        data = np.reshape(data, position_space.shape[0] * position_space.shape[1])
        data_3D = np.zeros([32,32,3])
        data_3D[:,:,0] = np.reshape(data, [32, 32])
        data_3D[:,:,1] = np.reshape(data, [32, 32])
        data_3D[:,:,2] = np.reshape(data, [32, 32])
        data = data_3D
    else:
        data = data[50 - np.int(np.sqrt(position_space.shape[0])) - 10:50 - 10,
               50 - np.int(np.sqrt(position_space.shape[0])) - 10:50 - 10]
        data = np.reshape(data, position_space.shape[0])

    data = np.array(data)
    
    # Restore original plotting settings as these were overwritten by plt.ioff()
    plt.close()
    plt.ion()
    mpl.rcParams['figure.dpi']= 200
    mpl.rcParams['font.size'] = 9.0
 
    mask = ift.Field.from_raw(position_space, data)
    M = ift.DiagonalOperator(mask)
    return M
