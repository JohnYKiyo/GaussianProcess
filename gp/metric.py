from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import vmap,jit
from functools import partial

def euclid_distance(x,y, square=True):
    '''
    \sum_m (X_m - Y_m)^2
    '''
    XX=np.dot(x.T,x)
    YY=np.dot(y.T,y)
    XY=np.dot(x.T,y)
    YX=np.dot(y.T,x)
    if not square:
        return np.sqrt(XX+YY-XY-YX)
    return XX+YY-XY-YX

def pairwise(dist,**kwargs):
    '''
    d_ij = dist(X_i , Y_j)
    "i,j" are assumed to indicate the data index.
    '''
    return jit(vmap(vmap(partial(dist,**kwargs),in_axes=(None,0)),in_axes=(0,None)))