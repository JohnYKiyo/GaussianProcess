from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit

from ..metric import euclid_distance
from ..utils import pairwise
from .base_kernel import BaseKernel

@jit
def gaussian_rbf(x1, x2, h=1.0, a=1.0):
    # distance between each rows
    dist = pairwise(euclid_distance,square=True)
    return a / np.sqrt(2 * np.pi * h**2) * np.exp(-0.5*dist(x1,x2) / h**2)

class GaussianRBFKernel(BaseKernel):
    def __init__(self,h=1.0,a=1.0,*args,**kwargs):
        self.__h = h
        self.__a = a

    def __call__(self,x1,x2):
        return gaussian_rbf(x1,x2,self.__h,self.__a)

    @property
    def h(self):
        return self.__h

    @property
    def a(self):
        return self.__a