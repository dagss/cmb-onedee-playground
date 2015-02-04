Simple example of constrained realization
=========================================

Instead of spherical harmonic we use the Discrete Cosine Transform.
In the generated plot you have, from the top:

 - Original signal
 - Signal after blurring it (increase beam_param for more blurring)
 - Signal after adding noise and masking out (increase sigma for more noise)
 - Reconstructed signal

If constrained_realization is True, "missing data" will be filled in
using a random realization consistent with the data model. Note that
this information-filling happens outside the mask too (less in areas
with less noise). Run script multiple times in an iPython session to
see how the reconstructed signal fluctuates under the mask.

To find the single most likely signal/mean, set constrained_realization=False.
This is equivalent to the average of infinitely many samples...and lacks
power compared to the prior power spectrum Cl.