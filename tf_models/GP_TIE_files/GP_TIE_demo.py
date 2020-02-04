
import numpy as np
from numpy import fft
import tensorflow as tf
from tensorflow import signal
from tensorflow import image
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from scipy.io import loadmat
import time

test_path = 'phase_rec_GUI/datasets/moustache_man_stack.mat'

data = loadmat(test_path)
Ividmeas = data['Istack']
z_vec = data['zvec']
if z_vec.shape[0] == 1:
    z_vec = z_vec.T

lambd = data['lambda'][0][0]
ps = data['ps']
zfocus = 1
Nsl = 50
eps1 = 1
eps2 = 1
reflect = False

reloaded_gptie = tf.saved_model.load("gptie")
start = time.time()
phase = reloaded_gptie(Ividmeas, z_vec, lambd, ps, zfocus, Nsl, eps1, eps2, reflect).numpy()
end = time.time()
print("time: ", end - start)
plt.imshow(phase)
plt.show()
