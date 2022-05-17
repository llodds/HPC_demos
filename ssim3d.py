# 3D SSIM using CuPy on GPU, could potentially extend to 3D

import cupy as cp
from cupyx.scipy.ndimage import uniform_filter

def ssim3D_cupy(im1, im2, win_size = 7, data_range = 2, full = False):
    """
    Compute 3D SSIM between two images (im1 and im2) using CuPy
    
    Parameters
    ----------
    im1, im2   : 3darray images
    win_size   : must be odd
    data_range : 2 for float32 (dmax = 1, dmin = -1)
    full       : return full SSIM matrix
    
    """
    
    # set parameters
    K1 = 0.01
    K2 = 0.03
    
    if win_size is None:
        win_size = 7
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd')

    NP = win_size ** 3
    cov_norm = NP / (NP - 1)
    
    # compute (weighted) means
    filter_func = uniform_filter
    filter_args = {'size': win_size}
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)
    
    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    
    R = data_range
    C1 = (K1*R) ** 2
    C2 = (K2*R) ** 2
    
    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    
    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2
    
    # compute (weighted) mean of ssim. Use float64 for accuracy.
    S2 = S[pad:S.shape[0]-pad, pad:S.shape[1]-pad, pad:S.shape[2]-pad]
    mssim = S2.mean(dtype=cp.float64)
    
    if full:
        return mssim, S
    else:
        return mssim
