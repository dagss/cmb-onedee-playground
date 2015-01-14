import numpy as np
from matplotlib.pyplot import *

"""
Problem:


"""



rng = np.random.RandomState(2)

# It's convenient to deal with real arrays instead of complex arrays, so
# we use a different FFT convention storing the harmonic signal as
# [Im(x_{n-1}) ... Im(x_1) Re(x_0) Re(x_1) ... Re(x_n)
#
# This repacking would have been a unitary transform if we had introduced
# some scaling (which I'm too lazy to find now), so it's OK from a linear
# algebra perspective.

def irfft(x):
    mid = (x.shape[0] - 1) // 2
    repacked_x = x[mid:].copy().astype(np.complex)
    repacked_x[1:-1] += 1j * x[:mid][::-1]
    return np.fft.irfft(repacked_x)

def rfft(x):
    y = np.fft.rfft(x)
    return np.concatenate([y.imag[1:][::-1], y.real[:-1]])

#
# Utilities
#
def hammer(matvec_func, n):
    u = np.zeros(n)
    out = np.zeros((n, n))
    for i in range(n):
        u[i] = 1
        out[:, i] = matvec_func(u)
        u[i] = 0
    return out

#
# Load the power spectrum used in the term S^{-1}, and the low-pass filter B
#
def load_Cl(lmax):
    dat = np.loadtxt('cl_beam_corbeam.bestfit_cl.dat')
    assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
    Cl = dat[:, 1][:lmax + 1]
    ls = np.arange(2, lmax + 1)
    Cl[2:] /= ls * (ls + 1) / 2 / np.pi
    Cl[0] = Cl[1] = Cl[2]
    return Cl

lmax = 512 + 1
Cl = load_Cl(lmax - 1)
bl = np.loadtxt('beam.txt')[:lmax]

packed_Cl = np.concatenate([Cl[1:-1][::-1], Cl])
packed_bl = np.concatenate([bl[1:-1][::-1], bl])

#
# Generate a fake noise map in pixel domain
#
npix = 1024
noise_standard_deviation = 0.04
Ninv = np.ones(npix) / noise_standard_deviation**2

# Mask out parts of it...

mask = np.ones(npix)
mask[100:200] = 0  # one big chunk masked out
mask[800:850] = 0  # one big chunk masked out

Ninv *= mask

# Generate fake data
true_signal_harmonic = rng.normal(size=packed_Cl.shape) * np.sqrt(packed_Cl)
true_signal_harmonic[0] = 0
noise = noise_standard_deviation * rng.normal(size=npix)

# Now compute:
# d = Y B s  ;
#    where Y is inverse FFT
#    B is low-pass filter that blurs image
#    s is original signal
#    d is observed data

data = irfft(packed_bl * true_signal_harmonic) + noise
data *= mask

#
# Problem all set up. Now reconstruct original image by brute-force.
# First we define the matrix-vector function; action of applying A
# if problem is A x = b
#

Y = hammer(irfft, npix)

def matvec(x_in):
    # A = S^{-1} + B Y^T N^{-1} Y B
    x = x_in * packed_bl # B
    x_pix = irfft(x) # Y
    x_pix *= Ninv # N^{-1}, a diagonal matrix
    # Y^T could be 'rfft' with the introduction of some scaling, but I'm too
    # lazy to find the appropriate scaling for Fourier transforms now, and
    # simply use a dense matrix-vector multiplication that doesn't scale well
    x = np.dot(Y.T, x_pix)  # Y^T
    x *= packed_bl # B
    return x + x_in / packed_Cl  # x + S^{-1} x_in

# Solve system densely
A = hammer(matvec, npix)
b = packed_bl * np.dot(Y.T, data)  # right-hand side: B Y^T d
solution = np.linalg.solve(A, b)  # to be done with iterative method...


# Plot our simulation so far
clf()
fig = gcf()
axs = [fig.add_subplot(4, 1, i) for i in range(1, 5)]

# Original "CMB"
axs[0].plot(irfft(true_signal_harmonic))

# After blurring by instrument
axs[1].plot(irfft(packed_bl * true_signal_harmonic))

# Adding noise and masking
axs[2].plot(data)

# Adding noise and masking
axs[3].plot(irfft(solution))


draw()
