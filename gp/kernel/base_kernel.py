from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit

class BaseKernel(object):
    def __init__(self):
        pass

    def __add__(self,other):
        return lambda x,y: self.__call__(x,y) + other.__call__(x,y)

    def __sub__(self,other):
        return lambda x,y: self.__call__(x,y) - other.__call__(x,y)