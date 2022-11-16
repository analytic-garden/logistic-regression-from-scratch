#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lr.py - a class for binary logistic regression
author: Bill Thompson
license: GPL 3
copyright: 2020-11-05
"""
import numpy as np
from scipy.linalg import norm
from scipy.special import expit  # logistic function

class lr(object):
    def __init__(self, 
            maxiter = 1000, 
            tol = 1.0e-06, 
            learn_rate = 0.01,
            debug = False):
        """
        Initialize object instance and set some parameters.
        Parameters
        ----------
        maxiter : int, optional
            maximum number of iteratiosn for fit, by default 1000
        tol : float, optional
            exit fit when gradient is less than tol, by default 1.0e-06
        learn_rate : float, optional
            learning rate, decrease coefficients by this times gradient, by default 0.01
        debug : bool, optional
            print debugging information, by default False
        """            
        self.maxiter = maxiter
        self.tol = tol
        self.learn_rate = learn_rate
        self.debug = debug

    def probability(self, X):
        """
        Calculate P(Y = 1|X, w)

        Parameters
        ----------
        X : numpy array
            n X m data matrix of observations

        Returns
        -------
        numpy array n X 1
            1/(1 + exp(-Xw)),
        """        
        return expit(np.matmul(X, self.weights))

    def predict(self, X):
        """
        Calculate predictions P(Y = 1|X, w) > 0.5

        Parameters
        ----------
        X : numpy array
            n X m data matrix of observations

        Returns
        -------
        numpy array
            0/1 prediction of y
        """        
        return np.rint(self.probability(X)).astype(np.int)

    def score(self, X, y):
        """
        Calculate number of difference between y and current prediction

        Parameters
        ----------
        X : numpy array
            n X m data matrix of observations
        y : numpy array
            n X 1 dependent variable from data CSV

        Returns
        -------
        int
            number of differences between predicted labels and data labels
        """        
        return self._diffs(y, self.predict(X))

    def _diffs(self, y, y_hat):
        """
        Calculate number of difference between y and current prediction

        Parameters
        ----------
        y : numpy array
            n X 1 dependent variable from data CSV
        y_hat : numpy array
            current prediction of P(Y = 1|X, w)

        Returns
        -------
        float
            number of differences between predicted labels and data labels
        """        
        labs = np.rint(y_hat).astype(np.int) # predicted labels
        return np.sum(np.abs(y - labs))

    @property
    def weights(self):
        """
        return current coefficients

        Returns
        -------
        numpy array
            m X 1 array of coefficients
        """        
        return self._w

    def log_likelihood(self, X, y):
        """
        Calculate negative log likelihood of data given current coefficients

        Parameters
        ----------
        X : numpy array
            n X m data matrix of observations
        y : numpy array
            n X 1 dependent variable from data CSV

        Returns
        -------
        float
            negative log likelihood of data given current coefficients
        """        
        p = self.probability(X)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        """
        Fit y ~ X

        Parameters
        ----------
        X : numpy array
            n X m data matrix of observations
        y : numpy array
            n X 1 dependent variable from data CSV

        Returns
        -------
        lr object
            this object. coefficients can be obtained from weight method.
        """        
        m, n = X.shape

        self._w = np.zeros((n, 1))   # coefficents 
        
        iter = 0
        while iter < self.maxiter: 
            y_hat = self.probability(X)   # current estimate
            grad = np.matmul(X.transpose(), (y_hat - y)) / m  # gradient
            self._w -= self.learn_rate * grad  # update coefficients

            if self.debug and (iter % 100 == 0):
                print(iter)
                print(self.weights)
                print(self._diffs(y, y_hat))
                print(grad, norm(grad)**2)
                print()

            # quit if gradient is flat
            if norm(grad) ** 2 < self.tol:
                break

            iter += 1
            
        return self
