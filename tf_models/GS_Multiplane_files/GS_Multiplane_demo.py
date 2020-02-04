import numpy as np
from numpy import fft
import tensorflow as tf
from tensorflow import signal
from tensorflow import image
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

test_path = 'moustache_man_stack.mat'

data = loadmat(test_path)
Ividmeas = data['Istack']
z_vec = data['zvec']
if z_vec.shape[0] == 1:
    z_vec = z_vec.T

lambd = data['lambda'][0][0]
ps = data['ps']
zfocus = 1
num_imgs = 1
Nsl = 50

reloaded_gs = tf.saved_model.load("gs_multiplane")
start = time.time()
phase = reloaded_gs(Ividmeas, zfocus, z_vec, num_imgs, ps, lambd, Nsl).numpy()
end = time.time()
print("time: ", end - start)
plt.imshow(phase)
plt.show()
