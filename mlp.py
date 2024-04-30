import numpy as np
from losses import *
from layers import *

class MLP:
    """
    A class for representing MLPs, which is amenable to performing
    backpropagation quickly using numpy operations

    Parameters
    ----------
    d: int
        Dimensions of the input
    est_lossderiv: function ndarray(N) -> ndarray(N)
        Gradient of the loss function with respect to the inputs
        to the last layer, using the output of the last layer
    """
    def __init__(self, d, est_lossderiv):
        ## TODO: Fill this in
        self.d = d
        self.est_lossderiv = est_lossderiv
        self.layers = []

        
        
    
    def add_layer(self, m, f, fderiv, name=None):
        """
        Parameters
        ----------
        m: int
            Number of neurons in the layer
        f: function ndarray(N) -> ndarray(N)
            Activation function, which is applied element-wise
        fderiv: function ndarray(N) -> ndarray(N)
            Derivative of activation function, which is applied element-wise
        name: string
            If specified, store the name of this layer
        """
        ## TODO: Fill this in
        n = self.d if len(self.layers) == 0 else self.layers[-1][0].shape[0]
        # find number of neurons in previous layer
        W = np.random.randn(m, n)*0.1
        b = np.random.randn(m)
        self.layers.append([W, b, f, fderiv, name, W*0, b*0])

    
    def forward(self, x, start=None, end=None):
        """
        Do a forward pass on the network, remembering the intermediate outputs
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        start: string
            If specified, start by feeding x to the input of this layer
        end: string
            If specified, stop and return the output of this layer
        
        Returns
        -------
        ndarray(m)
            Output of the network
        """
        ## TODO: Fill this in
        # find indices of start and end

        
        start_idx = 0
        end_idx = len(self.layers)-1
        if start:
            for i, layer in enumerate(self.layers):
                if start == layer[4]:
                    start_idx = i
        if end:
            for i, layer in enumerate(self.layers):
                if end == layer[4]:
                    end_idx = i
        
        L = len(self.layers)
        self.Hs = []
        self.As = []
        self.Hs.append(x.T) 
        for [W, b, f, fderiv, name, W_derivs, b_derivs] in self.layers[start_idx:end_idx+1]:
            hk1 = self.Hs[-1]
            a = W.dot(hk1) + b
            self.As.append(a)
            h = f(a)
            self.Hs.append(h)
        
        return self.Hs[-1]

    
    
    def backward(self, x, y):
        """
        Do backpropagation to accumulate the gradient of
        all parameters on a single example
        
        Parameters
        ----------
        x: ndarray(d)
            Input to feed through
        y: float or ndarray(k)
            Goal output.  Dimensionality should match dimensionality
            of the last output layer
        """
        ## TODO: Fill this in to complete backpropagation and accumulate derivatives
        L = len(self.layers)
        y_est = self.forward(x)
        g = self.est_lossderiv(y_est, y)

        for k in range(L-1, -1, -1):
            if k < L-1:
            # propagate gradient backwards through nonlinear output
                f_deriv = self.layers[k][3]
                g *= f_deriv(self.As[k])
            # compute gradients of weights and biases
            self.layers[k][6] += g
            #current_h = np.reshape(self.Hs[k-1], (1, len(self.Hs[k-1])))
            self.layers[k][5] += g[:, None]*self.Hs[k][None, :]
            # propagate gradient backwards through linear layer
            g = self.layers[k][0].T.dot(g) 
            

            

    def step(self, alpha):
        """
        Apply the gradient and take a step back

        Parameters
        ----------
        alpha: float
            Learning rate
        """
        ## TODO: Fill this in
        for i in range(len(self.layers)):
            self.layers[i][0] -= self.layers[i][5]*alpha
            self.layers[i][1] -= self.layers[i][6]*alpha

    def zero_grad(self):
        """
        Reset all gradients to zero
        """
        ## TODO: Fill this in
        for i in range(len(self.layers)):
            self.layers[i][5] *= 0 # weights
            self.layers[i][6] *= 0 # biases