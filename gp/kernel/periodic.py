from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit

from ..metric import euclid_distance, pairwise
from .base_kernel import BaseKernel

@jit
def periodic(x1, x2, a=1.0, h=1.0, p=1.0):
    # distance between each rows
    # aka periodic kernel (Exp-Sin-Squared kernel)
    dist = pairwise(euclid_distance,square=False)
    Sin = -2*np.square(np.sin(0.5*dist(x1,x2)/p))
    return a * np.exp(Sin / h**2)

class PeriodicKernel(BaseKernel):
    def __init__(self, a=1.0, h=1.0, p=1.0, *args,**kwargs):
        self.__a = a
        self.__h = h
        self.__p = p

    def __call__(self,x1,x2):
        return periodic(x1,x2, self.__a, self.__h, self.__p)

    @property
    def a(self):
        return self.__a

    @property
    def h(self):
        return self.__h

    @property
    def p(self):
        return self.__p