#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logreg.py - Binary logistic regression.
author: Bill Thompson
license: GPL 3
copyright: 2022-10-16
"""
import argparse
import sys

import numpy as np
import pandas as pd

from lr import lr

def GetArgs():
    def ParseArgs(parser):
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write('error: %s\n' % message)
                self.print_help()
                sys.exit(2)

        parser = Parser(description ='Logistic Regression')
        parser.add_argument('data_file',
                            help = 'Data CSV file for binary logistic regression. Dependent 0/1 variable must be in first column')
        parser.add_argument('-d', '--debug',
                            required=False,
                            default = False,
                            action = 'store_true',
                            help='Print debugging infomation during fit. Default = False')
        parser.add_argument('-m', '--max_iter',
                            required=False,
                            type=int,
                            default = 1000,
                            help='Maximum number of iterations. Default = 1000')
        parser.add_argument('-r', '--learning_rate',
                            required = False,
                            type = float,
                            default = 0.01,
                            help = 'Learning rate. Default = 0.01')
        parser.add_argument('-t', '--tolerance',
                            required=False,
                            type = float,
                            default = 0.001,
                            help='Tolerance for norm of gradient. Default = 0.001')
        
        return parser.parse_args()

    parser = argparse.ArgumentParser()
    args = ParseArgs(parser)

    return args

def get_data(data):
    """
    Separate data into X and y. y variable must be 0/1 and in first column.
    A column of 1'w is added to X to act as intercept.

    Parameters
    ----------
    data : Pandas dataframe
        A data frame with labeled columns.

    Returns
    -------
    a tuple
        X, y. X is an n X m numpy array. y is an n x 1 numpy array.
        n is the number of observations. m is the number of dependent variables
    """    
    X = data.iloc[:, 1:].to_numpy()
    X = np.hstack((np.ones([X.shape[0], 1]), X))
    y = data.iloc[:, 0].to_numpy()
    y = np.expand_dims(y, axis = 1)

    return X, y

def main():
    args = GetArgs()

    data = pd.read_csv(args.data_file)
    X, y = get_data(data)

    reg = lr(maxiter = args.max_iter, 
            tol = args.tolerance, 
            learn_rate = args.learning_rate,
            debug = args.debug)
    reg.fit(X, y)

    print(reg.weights, reg.score(X, y))

if __name__ == "__main__":
    main()
