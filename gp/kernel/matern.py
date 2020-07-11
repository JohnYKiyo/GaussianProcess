from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit

from ..metric import euclid_distance
from ..utils import pairwise
from .base_kernel import BaseKernel

@jit
def matern(*args,**kwargs):
    '''
    z=d(x_i,x_j)
    2(\sqrt{nu},z)**nu * Bessel_nu * 2(\sqrt{nu},z)**nu / Gamma(nu)
    '''
    raise NotImplementedError

class MaternKernel(BaseKernel):
    def __init__(self, a=1.0, h=1.0, p=1.0, *args,**kwargs):
        raise NotImplementedError

    def __call__(self,x1,x2):
        return matern()

    @property
    def a(self):
        return self.__a

    @property
    def h(self):
        return self.__h

    @property
    def p(self):
        return self.__p