'''cv_levelset.py

Implementation of the Chan Vese level set segmentation algorithm proposed in the
paper "Active Contours Without Edges - Chan, Vese - 2001" for grayscale 2D image data
'''
import time
import sys
import os
import math
import PIL
from PIL import Image
import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import helpers_cv_levelset as helpers
if __name__ != '__main__':
    raise ImportError('This module cannot be imported. Exiting with status (3)')

#################################################################
# load image from argv
if len(sys.argv) < 2:
    sys.exit('ArgumentError: Must supply image as an argument\n'
             '  Usage:  {!s} <image-file>'.format(sys.argv[0]))

p_image = sys.argv[1]
p_figs = os.path.join('figs', os.path.splitext(os.path.basename(p_image))[0])
os.makedirs(p_figs, exist_ok=True)
with Image.open(p_image) as image:
    im = np.array(image)
M, N = im.shape
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(im, cmap='gray')

#################################################################
# settings
verbose = True
save_figs = True
output_filetype = '.png'

# Objective Functional Term Weights
mu = 0.2              # contour length regularizer
nu = 0                # contour area regularizer
lambda1 = lambda2 = 1 # two-term data fidelities

                      # Hyperparams
dt = 0.5              # iteration time step
eps = 1               # Heaviside/Delta regularizer
eta = 10e-8           # prevent div-by-0 errors

tol = 1e-3            # stopping criteria
max_iters = 20        # halting criteria

#################################################################
# closures - for shortening lsf update function
def A(lsf, i, j):
    return helpers.coeff_A(lsf, i, j, mu, eta)

def B(lsf, i, j):
    return helpers.coeff_B(lsf, i, j, mu, eta)

def H(t):
    return helpers.heaviside_reg(t, eps)

def D(t):
    return helpers.delta_reg(t, eps)

#################################################################
# initialize lsf to checkerboard
lsf = np.zeros(im.shape)
for x1 in range(lsf.shape[0]):
    for x2 in range(lsf.shape[1]):
        lsf[x1, x2] = math.sin(math.pi/5*x1) * math.sin(math.pi/5*x2)
I = helpers.BoundsHandler_Clamp(im)
L = helpers.BoundsHandler_Clamp(lsf)

iter = 0
break_loop = False
while True:
    iter += 1
    sys.stdout.write('iter: {:4d}'.format(iter))

    # keep current lsf copy for tolerance check
    lsf_n = np.copy(lsf)

    # compute c1 and c2 as region averages
    #  c1, c2 = helpers.update_C(I, L)
    c1, c2 = helpers.update_C_reg(I, L, H)

    # evolve lsf by one timestep (dt)
    # only in narrowband
    for i in range(M):
        for j in range(N):
            dtd = dt*D(L[i,j])
            L[i,j] = (L[i,j] + dtd*(A(L,i,j)*L[i+1,j] + A(L,i-1,j)*L[i-1,j] +
                                    B(L,i,j)*L[i,j+1] + B(L,i,j-1)*L[i,j-1] -
                                    nu - lambda1*(I[i,j]-c1)**2 + lambda2*(I[i,j]-c2)**2)
                     ) / (1 + dtd*(A(L,i,j)+A(L,i-1,j)+B(L,i,j)+B(L,i,j-1)))

    # check stopping criteria and report the iteration
    if break_loop or save_figs:
        helpers.update_fig_contour(ax, lsf)
        fig_filename = 'iter_{:04d}'.format(iter)
    err = np.linalg.norm(lsf.ravel() - lsf_n.ravel(), 2) / (M*N)
    if verbose:
        sys.stdout.write(' || C1: {:7.2f}, C2: {:7.2f}'.format(c1, c2))
        sys.stdout.write(' | cost: {:10.4e}\n'.format(err))
    if err < tol:
        print('\nCONVERGED  final cost: {:f}'.format(err))
        ax.set_title('Convergence after {:d} iters'.format(iter))
        fig_filename += '_convergence'
        break_loop = True
    elif iter >= max_iters:
        print('\nHALT  exceeded max iterations')
        ax.set_title('Early halt after {:d} iters'.format(iter))
        fig_filename += '_earlyhalt'
        break_loop = True
    if break_loop:
        fig.savefig(os.path.join(p_figs, fig_filename + output_filetype))
        break

    # prepare for next iter
    if save_figs:
        ax.set_title('iter {:4d}'.format(iter))
        fig.savefig(os.path.join(p_figs, fig_filename + output_filetype))

    # optionally reinit lsf with signed distance function every N iters
# END WHILE

plt.show()
