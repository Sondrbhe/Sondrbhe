# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:32:37 2020

@author: iver_

This library is based on Jean Jordanou "RNN" library

__init__ will initiate all random reservoir.
"""

import numpy as np
import scipy.io as io

class EchoStateNetwork:
    
    def __init__(self, nodes, input_size, output_size,
                 leak_rate=0.5, spectral_radius=1, psi=0.5, input_scaling=0.1,
                 bias_scaling =0.5, alpha = 10, forget_factor = 1,
                 output_feedback = False,
                 output_scaling = 0,
                 noise_amp = 0):
        print("Initializing reservoir")
        # Basic parameters
        self.nodes = nodes
        self.input_size = input_size
        self.output_size = output_size
        
        self.psi = psi # Sparsity (0-1)
        # Reservoir matrix (random)
        self.Wrr0 = np.random.normal(0,1,[nodes,nodes])
        #self.Wrr0 = sparsity(self.Wrr0, self.psi) # Sparsity (0-1) is how many cells in matrix that are populated (1 -> empty matrix)
        
        # Weighting matrices (random) for input, output and bias.
        self.Wir0 = np.random.normal(0,1,[nodes, input_size])
        self.Wbr0 = np.random.normal(0,1,[nodes, 1])
        self.Wor0 = np.random.normal(0,1,[nodes, output_size])
        
        self.Wro = np.random.normal(0,1,[output_size, nodes+1]) # USikker på denne atm.
        
        # ESN parameters
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        
        # Recursive Leas Squares (RLS) algorithm parameters
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.output_feedback = output_feedback # Don't think we need this one. Feedback should not be necessary
        
        # For warmup
        self.a = np.zeros([nodes, 1],dtype=np.float64)
        
        self.Wrr = self.Wrr0
        
        # Making Rho the maximum eigenvalue equal to the spectral radius (rho(W)))
        eigenvalues = np.linalg.eigvals(self.Wrr)
        radius = np.abs(np.max(eigenvalues))
        self.Wrr = self.Wrr/radius
        self.Wrr *= spectral_radius
        
        
        # Matrix scaling
        self.Wbr = bias_scaling*self.Wbr0
        self.Wir = input_scaling*self.Wir0
        self.Wor = output_scaling*self.Wor0
        self.noise = noise_amp
        
        # Covariance matrix
        self.P = np.eye(nodes+1)/alpha
        
    def off_training(self, X, Y, regularization, warmup): # X is inputs, Y is desired outputs.
        A = np.empty([Y.shape[0]-warmup,self.nodes])
        for i in range(Y.shape[0]):
            # Output feedback uses need feedback ofc (Won't be used in the first place)
            if self.output_feedback:
                if i > 0:
                    self.update(X[i,:],Y[i-1,:])
                else:
                    self.update(X[i,:])
                    
            else:
                self.update(X[i,:])
                if i > warmup:
                    A[i - warmup, :] = self.a.T
                    
        A_wbias = np.hstack((np.ones([A.shape[0],1]), A))
        # Dot product, må komme tilbake til den her
        P = np.dot(A_wbias.T, A_wbias) # X^T*X
        R = np.dot(A_wbias.T,Y[warmup:]) # X^2*Y_estimated?
        
        # Ridge regression, the regularization should be tested with different values, eg. logaritmic
        Theta = np.linalg.solve(P+regularization*np.eye(self.nodes+1,self.nodes+1),R)
        self.Wro = Theta.T
        
        
    def update(self, input1, y_in = np.atleast_2d(0)):
        Input = np.array(input1)
        Input = Input.reshape(Input.size,1)
        Y_in = np.array(y_in)
        Y_in = Y_in.reshape(Y_in.size,1)
        
        if (y_in == 0).all():
            Y_in = np.zeros([self.output_size,1])
            
        if Input.size == self.input_size:
            z = np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input)+self.Wbr
            if self.output_feedback:
                z += np.dot(self.Wor, Y_in)
            if self.noise > 0:
                z += np.random.normal(0, self.noise, [self.nodes, 1])
            self.a = (1-self.leak_rate)*self.a + self.leak_rate*np.tanh(z)
            
            a_wbias = np.vstack((np.atleast_2d(1.0), self.a))
            y = np.dot(self.Wro, a_wbias)
            return y
        else:
            raise ValueError("Input must have the same size as input_size")
            
            
    def reset(self):
        self.a = np.zeros([self.nodes, 1])
            