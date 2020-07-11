from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import vmap,jit

def euclid_distance(x,y, square=True):
    '''
    \sum_m (X_m - Y_m)^2
    '''
    XX=np.dot(x.T,x)
    YY=np.dot(y.T,y)
    XY=np.dot(x.T,y)
    if not square:
        return np.sqrt(XX+YY-2*XY)
    return XX+YY-2*XY