import numpy as np

def logistic(u):
    """
    Compute the logistic function on a set of inputs

    Parameters
    ----------
    u: ndarray(N)
        A set of inputs to the logistic function
    
    Returns
    -------
    ndarray(N): logistic function outputs
    """
    return 1/(1+np.exp(-u))

def logistic_deriv(u):
    """
    Compute the logistic function's derivative on a set of inputs

    Parameters
    ----------
    u: ndarray(N)
        A set of inputs to the logistic function
    
    Returns
    -------
    ndarray(N): Derivatives of logistic function at these input
    """
    res = logistic(u)
    return res*(1-res)

def leaky_relu(u, alpha=0.01):
    """
    Compute the leaky ReLU on a set of inputs

    Parameters
    ----------
    u: ndarray(N)
        A set of inputs to the Leaky ReLU function
    alpha: float
        Slope in negative region
    
    Returns
    -------
    ndarray(N): Leaky ReLU outputs
    """
    res = np.zeros_like(u)
    res[u > 0] = u[u > 0]
    res[u < 0] = alpha*res[u < 0]
    return res 

def leaky_relu_deriv(u, alpha=0.01):
    """
    Compute the leaky ReLU's derivative on a set of inputs

    Parameters
    ----------
    u: ndarray(N)
        A set of inputs to the Leaky ReLU
    
    Returns
    -------
    ndarray(N): Derivatives of Leaky ReLU at these inputs
    """
    res = np.zeros_like(u)
    res[u > 0] = 1
    res[u < 0] = alpha
    return res

def softmax(u):
    """
    Compute the softmax of an input

    Parameters
    ----------
    u: ndarray(N)
        Inputs to softmax
    
    Returns
    -------
    ndarray(N)
        The softmax of the input at every dimension
    """
    res = np.exp(u)
    return res/np.sum(res)