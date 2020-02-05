### Referenced from phase_rec_GUI
import numpy as np
from numpy import fft
import tensorflow as tf
from tensorflow import signal
from tensorflow import image
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

class GS_Multiplane(tf.Module):

    def __init__(self):
        self.center_prop = None
        self.Eout = None
        return

    @tf.function
    def __call__(self, Inten, I0_idx, zvec, num_imgs, ps, lambd, N):
        ps = tf.cast(ps, tf.float32)
        lambd = tf.cast(lambd, tf.float32)
        zvec = tf.cast(zvec, tf.float32)
        Inten = tf.cast(Inten, tf.complex64)
        (n, m, _) = Inten.shape

        # might have to do I0_idx-1.  Also what's the point of 1j * zeros?
        E0_est = tf.sqrt(Inten[:,:,I0_idx]) * tf.exp(1j * tf.zeros((n,m), dtype = tf.complex64))

        if self.center_prop is None:
            self.center_prop = tf.Variable(tf.zeros((n,m,1)))

        for iternum in range(N):

            # Propagate forward and replace with measured intensity
            prop = self.propagate(E0_est, lambd, zvec[I0_idx+1:I0_idx+num_imgs+1], ps)
            prop = tf.sqrt(Inten[:,:,I0_idx+1:I0_idx+num_imgs+1]) * tf.exp(1j * tf.cast(tf.math.angle(prop), tf.complex64))

            # Propagate all planes back to center
            for i in range(num_imgs):
                self.center_prop[:,:,i].assign(tf.squeeze(tf.math.angle(self.propagate(prop[:,:,i],lambd, -1 * zvec[I0_idx+i+1], ps))))
            # Replace with center plane intensity
            E0_est = tf.sqrt(Inten[:,:,I0_idx]) * tf.exp(1j * tf.cast(tf.reduce_mean(self.center_prop,2), tf.complex64))

            # Flip stack for negative propagations
            stk_temp = tf.reverse(Inten[:,:,I0_idx-num_imgs:I0_idx],axis=[2])
            z_temp = tf.reverse(zvec[I0_idx-num_imgs:I0_idx], axis=[0])

            # Propagate backward and replace with measured intensity
            prop = self.propagate(E0_est, lambd, z_temp, ps)
            prop = tf.sqrt(stk_temp) * tf.exp(1j * tf.cast(tf.math.angle(prop), tf.complex64))

            # Propagate all planes back to center
            for i in range(num_imgs):
                self.center_prop[:,:,i].assign(tf.squeeze(tf.math.angle(self.propagate(prop[:,:,i],lambd, -1 * z_temp[i], ps))))

            # Replace with center plane intensity
            E0_est = tf.sqrt(Inten[:,:,I0_idx]) * tf.exp(1j * tf.cast(tf.reduce_mean(self.center_prop,2), tf.complex64))

        phi = tf.math.angle(E0_est)
        return phi

    @tf.function
    def propagate(self, Ein, lambd, Z, ps): #, varargin):

        (m, n) = Ein.shape
        M = m
        N = n

        gpu_num = 0 # check gpu

        mask = 1


        # Initialize variables into CPU or GPU
        if (gpu_num == 0):
            if self.Eout is None:
                self.Eout = tf.Variable(1j * tf.zeros((m,n,len(Z)), dtype = tf.complex64))
            aveborder = tf.reduce_mean(tf.concat((Ein[0,:], Ein[m-1,:], tf.transpose(Ein[:,0]), tf.transpose(Ein[:,n-1])), axis=0))
            #np.mean(cat(2,Ein(1,:),Ein(m,:),Ein(:,1)',Ein(:,n)'));
            H = tf.zeros((M,N,len(Z)))

        else:
            # reset(gpuDevice(1));
            raise NotImplementedError
            # lambd = gpuArray(lambd);
            # Z = gpuArray(Z);
            # ps = gpuArray(ps);
            # Eout = gpuArray.zeros(m,n,length(Z));
            # aveborder=gpuArray(mean(cat(2,Ein(1,:),Ein(m,:),Ein(:,1)',Ein(:,n)')));
            # if nargout>1
            #     H = gpuArray.zeros(M,N,length(Z));


        # Spatial Sampling
        [x,y] = tf.meshgrid(tf.range(-N/2,(N/2-1)+1), tf.range(-M/2,(M/2-1)+1))

        fx = (x / (ps * M))    #frequency space width [1/m]
        fy = (y / (ps * N))    #frequency space height [1/m]
        fx2fy2 = fx ** 2 + fy ** 2


        # Padding value
        Ein_pad = Ein
        # Ein_pad = tf.ones((M,N), dtype = tf.complex64) * aveborder #pad by average border value to avoid sharp jumps
        # Ein_pad[(M-m)//2:(M+m)//2,(N-n)//2:(N+n)//2] = Ein # what is this?
        # Ein_pad = tf.pad(Ein, ((), ()), aveborder)

        # FFT of E0
        E0fft = signal.fftshift(signal.fft2d(Ein_pad))
        for z in range(len(Z)):
            H  = tf.exp(-1j * np.pi * tf.cast(lambd * Z[z] * fx2fy2, tf.complex64)) #Fast Transfer Function
            Eout_pad=signal.ifft2d(signal.ifftshift(E0fft * H * mask))

            self.Eout[:,:,z].assign(Eout_pad[(M-m)//2:(M+m)//2,(N-n)//2:(N+n)//2])
            # Eout[:,:,z]=Eout_pad[(M-m)//2:(M+m)//2,(N-n)//2:(N+n)//2]


        # Gather variables from GPU if necessary
        if (gpu_num > 0):
            raise NotImplementedError
            # Eout=gather(Eout);
            # if nargout > 1:
            #     H=gather(H);

        return self.Eout # H not returned?


test_path = 'phase_rec_GUI/datasets/moustache_man_stack.mat'

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


start = time.time()
gs_multiple_plane = GS_Multiplane()

phase = gs_multiple_plane(Ividmeas, zfocus, z_vec, num_imgs, ps, lambd, Nsl).numpy()
end = time.time()
print("time: ", end - start)
plt.imshow(phase)
plt.show()

tf.saved_model.save(gs_multiple_plane, "gs_multiplane")

reloaded_gs = tf.saved_model.load("gs_multiplane")
phase = reloaded_gs(Ividmeas, zfocus, z_vec, num_imgs, ps, lambd, Nsl).numpy()
plt.imshow(phase)
plt.show()
