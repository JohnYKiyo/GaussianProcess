from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import vmap,jit
from functools import partial

@jit
def transform_data(x):
    if isinstance(x,np.ndarray):
        if len(x.shape)==1:
            return np.atleast_2d(x.astype(np.float64)).T
        else:
            return np.atleast_2d(x.astype(np.float64))
    elif isinstance(x,list):
        return transform_data(np.array(x))
    else:
        raise ValueError("Cannot convert to numpy.array")

@jit
def data_checker(X,Y):
    '''N_data check'''
    if len(X) != len(Y):
        raise ValueError('X,Y should be same number of data. (n,d),(n,)')

def pairwise(dist,**kwargs):
    '''
    d_ij = dist(X_i , Y_j)
    "i,j" are assumed to indicate the data index.
    '''
    return jit(vmap(vmap(partial(dist,**kwargs),in_axes=(None,0)),in_axes=(0,None)))