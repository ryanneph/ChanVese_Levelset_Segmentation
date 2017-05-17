'''helpers_cv_levelset.py

Collection of functions use in the update iterations for the Chan Vese level set segmentation algorithm
proposed in the paper "Active Contours Without Edges - Chan, Vese - 2001"
'''
import math
import numpy as np

### FUNCTIONS #############################################################################################
def heaviside_reg(t, eps=1):
    '''Heaviside approximation in time
    Args:
        t (float):   time
        eps (float): regularization param
    '''
    return 0.5*(1 + 2/math.pi*math.atan2(t, eps))

def delta_reg(t, eps=1):
    '''derivative of Heaviside approximation in time
    Args:
        t (float):   time
        eps (float): regularization param
    '''
    return eps / (math.pi*(eps**2 + t**2))

def coeff_A(lsf, i, j, mu=0.2, eta=10e-8):
    return mu / math.sqrt(eta**2 + (lsf[i+1,j]-lsf[i,j])**2 + ((lsf[i,j+1]-lsf[i,j-1])/2)**2)

def coeff_B(lsf, i, j, mu=0.2, eta=10e-8):
    return mu / math.sqrt(eta**2 + ((lsf[i+1,j]-lsf[i-1,j])/2)**2 + (lsf[i,j]-lsf[i+1,j])**2)

def update_C(image, lsf):
    if isinstance(image, BoundsHandler_Base): image = image.array
    if isinstance(lsf, BoundsHandler_Base): lsf = lsf.array
    c1 = np.mean(image[lsf>0]) # inner region mean
    c2 = np.mean(image[lsf<0]) # outer region mean
    return (c1, c2)

def update_C_reg(image, lsf, selector=heaviside_reg):
    if isinstance(image, BoundsHandler_Base):
        M, N = image.array.shape
    else:
        M, N = image.shape

    c12 = [0, 0]
    for k in range(2):
        _c = 0
        _n = 0
        for i in range(M):
            for j in range(N):
                _c += image[i,j]*(k-selector(lsf[i,j]))
                _n += (k-selector(lsf[i,j]))
        c12[k] = _c/_n
        del _c, _n
    return tuple(c12)

def update_fig_contour(ax, lsf):
    if isinstance(lsf, BoundsHandler_Base): lsf = lsf.array
    # remove old contours first
    try:
        for c in ax.collections:
            c.remove()
    except: pass
    return ax.contour(lsf, levels=[0], colors=['red'])

### CLASSES ###############################################################################################
class BoundsHandler_Base():
    '''Base class that doesnt implement safe bounds handling and throws numpy error instead'''
    def __init__(self, arr):
        self.array = arr

    def _bounds_handler(self, idx):
        return idx

    def __getitem__(self, idx):
        return self.array[self._bounds_handler(idx)]

    def __setitem__(self, idx, val):
        self.array[self._bounds_handler(idx)] = val

class BoundsHandler_Clamp(BoundsHandler_Base):
    '''Clamping bounds:
    arr[-1, j] = arr[0,   j]
    arr[M,  j] = arr[M-1, j]
    arr[i, -1] = arr[i,   0]
    arr[i,  N] = arr[i, N-1]
    '''
    def __init__(self, arr):
        super().__init__(arr)

    def _bounds_handler(self, idx):
        idx = list(idx)
        for i in range(len(idx)):
            if idx[i] >= self.array.shape[i]: idx[i] = self.array.shape[i]-1
            if idx[i] < 0: idx[i] = 0
        return tuple(idx)

class BoundsHandler_Mirror(BoundsHandler_Base):
    '''Mirroring bounds:
    arr[-1, j] = arr[+1,  j]
    arr[M,  j] = arr[M-2, j]
    arr[i, -1] = arr[i,  +1]
    arr[i,  N] = arr[i, N-2]
    '''
    def __init__(self, arr):
        super().__init__(arr)

    def _bounds_handler(self, idx):
        idx = list(idx)
        for i in range(len(idx)):
            if idx[i] >= self.array.shape[i]: idx[i] = 2*self.array.shape[i] - idx[i] - 2
            if idx[i] < 0: idx[i] = -idx[i]
        return tuple(idx)
