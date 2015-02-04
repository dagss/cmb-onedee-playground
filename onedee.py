import numpy as np
from scipy.fftpack import dct, idct
from matplotlib.pyplot import *

#
# Parameters
#
constrained_realization = True
# ^^^ if False, it's a Wiener-filter. If True, we will fill in unknown information with
# a random sample that is consistent with our data model
npix = 512
sigma = 1  # standard deviation of per-pixel noise
beam_param = 0.01  # how much blurring to do

mask = np.ones(npix)
mask[100:200] = 0  # chunk masked out
mask[400:450] = 0  # chunk masked out
rms = 1 + 0.9 * np.sin(np.linspace(0, 2 * np.pi, npix))  # rms is "sigma per pixel", use a sin-function to make some parts have less noise than others
rms *= sigma

rng = np.random.RandomState(2)  # rng to set up fake data, can be static
rng_sample = np.random  # rng used to draw constrained realizations, should be different each run


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

lmax = npix - 1
Cl = load_Cl(lmax)
ls = np.arange(lmax + 1)
bl = np.exp(-0.5 * ls * (ls+1) * beam_param**2)

Ninv = 1 / rms**2
Ninv *= mask

# Generate fake data
true_signal_harmonic = rng.normal(size=Cl.shape) * np.sqrt(Cl)
true_signal_harmonic[0] = 0
noise = rms * rng.normal(size=npix)

# Now compute:
# d = Y B s  ;
#    where Y is inverse FFT
#    B is low-pass filter that blurs image
#    s is original signal
#    d is observed data

data = idct(bl * true_signal_harmonic) + noise
data *= mask

#
# Problem all set up. Now reconstruct original image by brute-force.
# First we define the matrix-vector function; action of applying A
# if problem is A x = b
#

Y = hammer(idct, npix)

def matvec(x_in):
    # A = S^{-1} + B Y^T N^{-1} Y B
    x = x_in * bl # B
    x_pix = idct(x) # Y
    x_pix *= Ninv # N^{-1}, a diagonal matrix

    # At this point we want "transpose idct", which differs from "dct" by a scale factor.
    # Just use the O(npix^2) multiplication for now... there should be some rescaling
    # combined with `dct` that makes this fast.
    x = np.dot(Y.T, x_pix)
    x *= bl # B
    return x + x_in / Cl  # x + S^{-1} x_in

# Solve system densely
A = hammer(matvec, npix)
b = bl * np.dot(Y.T, Ninv * data)  # right-hand side: B Y^T d

if constrained_realization:
    b += bl * np.dot(Y.T, np.sqrt(Ninv) * rng_sample.normal(size=npix))
    b += np.sqrt(1/Cl) * rng_sample.normal(size=lmax + 1)

solution = np.linalg.solve(A, b)  # to be done with iterative method...


# Plot our simulation so far
clf()
fig = gcf()
axs = [fig.add_subplot(4, 1, i) for i in range(1, 5)]

# Original "CMB"
axs[0].plot(idct(true_signal_harmonic))

# After blurring by instrument
axs[1].plot(idct(bl * true_signal_harmonic))

# Adding noise and masking
axs[2].plot(data)

# Adding noise and masking
axs[3].plot(idct(solution))

for ax, title in zip(axs, ['Signal', 'Blurred', 'With noise', 'Reconstructed']):
    ax.set_ylim((-300, 300))
    ax.set_ylabel(title)

draw()
