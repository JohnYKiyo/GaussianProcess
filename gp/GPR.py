from .kernel import GaussianRBFKernel
from .utils import transform_data,data_checker

from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import vmap,jit

class GPR(object):
    '''Gaussian Process Regression
    GPR is a nonparametric, Bayesian approach to regression.
    GPR provides uncertainty measurements on the predictions.

    Attributes:
        X_train (array-like) :
            array shape (n_samples, n_features) or (n_samples,)
            Feature vectors or other representations of training data (also required for prediction).
        Y_train (array-like) :
            array shape (n_samples,) or (n_samples, n_targets) )
            Target values in training data (also required for prediction)
        kernel:
    '''
    def __init__(self, X_train, Y_train,  *args, **kwargs):
        '''init
        Args:
            X_train (array-like) :
                array shape (n_samples, n_features) or (n_samples,)
                Feature vectors or other representations of training data (also required for prediction).

            Y_train (array-like) :
                array shape (n_samples,) or (n_samples, n_targets) )
                Target values in training data (also required for prediction.

            **alpha (float) :
                Regularization parameter. Defaults 0.

            **kernel (function or kernel instance) :
                The kernel used for prediction. The structure of the kernel is the same as the one passed as parameter but with optimized hyperparameters
                Defaults Gaussian RBF kernel h=0., a=1.
        '''

        self.__X_train = transform_data(X_train)
        self.__Y_train = transform_data(Y_train)
        data_checker(self.__X_train,self.__Y_train)

        self.__kernel = kwargs.pop('kernel',GaussianRBFKernel(h=1.0,a=1.0))
        self.__alpha = kwargs.pop('alpha',0)
        self.__jitter = 1e-6
        self.__kwargs = kwargs

        self.__compute_gram_matrix()

    def __compute_gram_matrix(self):
        self.__K = self.__kernel(self.__X_train, self.__X_train) +\
            (self.__alpha+self.__jitter) * np.eye(len(self.__X_train))
        self.__K_inv = np.linalg.inv(self.__K)

    def posterior_predictive(self,x,return_std=False,return_cov=False):
        '''posterior
        This methods provides predictions and uncertainties for inputs.

        Args:
            x (array_like):
                array shape (n_samples, n_features) or (n_samples,)

            return_std (bool):
                If true, include Standard deviation of predictions in the output.

            return_cov (bool):
                If true, include Covariance of predictions in the output.
        Returns:
            Value (array-like) : Predictions.
            Std (array-like) : Standard deviation of predictions each input x if return_std is True.
            Cov (array-like) : Covariance matrix of predictions if return_cov is True.
        '''
        X = transform_data(x)
        K_s = self.__kernel(self.__X_train, X)
        K_ss = self.__kernel(X, X)
        mu_s = np.dot(np.dot(K_s.T, self.__K_inv), self.__Y_train)
        cov_s = K_ss - np.dot(np.dot(K_s.T, self.__K_inv), K_s)

        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if return_cov:
            return mu_s, cov_s

        elif return_std:
            return mu_s, np.atleast_2d(np.sqrt(np.diag(cov_s))).T

        else:
            return mu_s

    def append_data(self,X_train,Y_train):
        '''Append method
        Append data to prediction.

        Args:
            X_train (array-like) :
                array shape (n_samples, n_features) or (n_samples,)
                Feature vectors or other representations of training data (also required for prediction).

            Y_train (array-like) :
                array shape (n_samples,) or (n_samples, n_targets) )
                Target values in training data (also required for prediction.
        '''
        X_new = transform_data(X_train)
        Y_new = transform_data(Y_train)
        data_checker(X_new,Y_new)
        '''check dim'''
        if X_new.shape[1] != self.__X_train.shape[1]:
            raise ValueError(f'X dimention is not correct. X_train:{self.__X_train.shape}')
        self.__X_train = np.row_stack([self.__X_train,X_new]) ##original numpy: np.r_([self.__X_train,X_new])
        self.__Y_train = np.row_stack([self.__Y_train,Y_new]) ##original numpy: np.r_([self.__Y_train,Y_new])
        self.__compute_gram_matrix()

    '''getter'''
    @property
    def X_train(self):
        return self.__X_train

    @property
    def Y_train(self):
        return self.__Y_train

    @property
    def kernel(self):
        return self.__kernel
