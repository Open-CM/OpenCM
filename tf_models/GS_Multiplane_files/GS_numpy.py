 ### Referenced from phase_rec_GUI
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

def gs_multiple_plane(Inten, I0_idx, zvec, num_imgs, ps, lambd, N):

    (n, m, _) = Inten.shape
    # might have to do I0_idx-1.  Also what's the point of 1j * zeros?
    E0_est = np.sqrt(Inten[:,:,I0_idx]) * np.exp(1j * np.zeros((n,m)))

    for iternum in range(N):

        # Propagate forward and replace with measured intensity
        prop = propagate(E0_est, lambd, zvec[I0_idx+1:I0_idx+num_imgs+1], ps)
        prop = np.sqrt(Inten[:,:,I0_idx+1:I0_idx+num_imgs+1]) * np.exp(1j * np.angle(prop))

        # Propagate all planes back to center
        for i in range(num_imgs):
            prop[:,:,i] = np.squeeze(np.angle(propagate(prop[:,:,i],lambd, -1 * zvec[I0_idx+i+1], ps)))

        # Replace with center plane intensity
        E0_est = np.sqrt(Inten[:,:,I0_idx]) * np.exp(1j * np.mean(prop,2))

        # Flip stack for negative propagation
        stk_temp = np.flip(Inten[:,:,I0_idx-num_imgs:I0_idx],2)
        z_temp = np.flip(zvec[I0_idx-num_imgs:I0_idx])

        # Propagate backward and replace with measured intensity
        prop = propagate(E0_est, lambd, z_temp, ps)
        prop = np.sqrt(stk_temp) * np.exp(1j * np.angle(prop))

        # Propagate all planes back to center
        for i in range(num_imgs):
            prop[:,:,i] = np.squeeze(np.angle(propagate(prop[:,:,i],lambd, -1 * z_temp[i], ps)))

        # Replace with center plane intensity
        E0_est = np.sqrt(Inten[:,:,I0_idx]) * np.exp(1j * np.mean(prop,2))

    phi = np.angle(E0_est)
    print("E0_est:", np.min(E0_est), np.max(E0_est))
    return phi

def propagate(Ein, lambd, Z, ps): #, varargin):

    (m, n) = Ein.shape
    M = m
    N = n

    gpu_num = 0 # check gpu

    mask = 1


    # Initialize variables into CPU or GPU
    if (gpu_num == 0):
        Eout = 1j * np.zeros((m,n,len(Z)))
        aveborder = np.mean(np.concatenate((Ein[0,:], Ein[m-1,:], Ein[:,0].T, Ein[:,n-1].T)))
        #np.mean(cat(2,Ein(1,:),Ein(m,:),Ein(:,1)',Ein(:,n)'));
        H = np.zeros((M,N,len(Z)))

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
    [x,y] = np.meshgrid(np.arange(-N/2,(N/2-1)+1), np.arange(-M/2,(M/2-1)+1))
    fx = (x / (ps * M))    #frequency space width [1/m]
    fy = (y / (ps * N))    #frequency space height [1/m]
    fx2fy2 = fx ** 2 + fy ** 2


    # Padding value
    Ein_pad=np.ones((M,N)) * aveborder #pad by average border value to avoid sharp jumps
    Ein_pad[(M-m)//2:(M+m)//2,(N-n)//2:(N+n)//2] = Ein # what is this?


    # FFT of E0
    E0fft = fft.fftshift(fft.fft2(Ein_pad))
    for z in range(len(Z)):
        H  = np.exp(-1j * np.pi * lambd * Z[z] * fx2fy2) #Fast Transfer Function
        Eout_pad=fft.ifft2(fft.ifftshift(E0fft * H * mask))

        Eout[:,:,z]=Eout_pad[(M-m)//2:(M+m)//2,(N-n)//2:(N+n)//2]


    # Gather variables from GPU if necessary
    if (gpu_num > 0):
        raise NotImplementedError
        # Eout=gather(Eout);
        # if nargout > 1:
        #     H=gather(H);

    return Eout # H not returned?


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
phase = gs_multiple_plane(Ividmeas, zfocus, z_vec, num_imgs, ps, lambd, Nsl)
end = time.time()
print("time: ", end - start)
print(phase)
print(np.min(phase), np.max(phase))
plt.imshow(phase)
plt.show()
