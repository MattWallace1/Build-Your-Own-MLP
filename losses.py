import numpy as np

def logistic_est_loss(y_est, y):
    """
    Compute the logistic loss

    Parameters
    ----------
    y_est: float
        Input to logistic function
    y: float
        Target value
    
    Returns
    -------
    float: Logistic loss over all samples
    """
    return -y*np.log(y_est) - (1-y)*np.log(1-y_est)


def logistic_loss_deriv(u, y):
    """
    Compute the gradient of the logistic loss over many samples
    with respect to its inputs

    Parameters
    ----------
    u: ndarray(N)
        Input to logistic function
    y: float
        Target value
    
    Returns
    -------
    float:
        Derivative of logistic loss over all samples
    """
    from layers import logistic
    return logistic(u)-y


def logistic_est_loss_deriv(y_est, y):
    """
    Compute the derivative of the logistic loss with
    respect to its input variable, given only the 
    output y_est of the logistic function

    Parameters
    ----------
    y_est: float
        Estimated output of the logistic function
    y: float
        Target value
    
    Returns
    -------
    float:
        Derivative of logistic loss with respect to input
        *u* to logistic function
    """
    return y_est-y

def softmax_est_crossentropy_loss(y_est, y):
    """
    Compute the multiclass softmax cross-entropy loss
    given the output of the softmax function

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the softmax function
    y: ndarray(N)
        Target values
    
    Returns
    -------
    float: Cross-entropy
    """
    return -np.sum(y*np.log(y_est))

def softmax_est_crossentropy_deriv(y_est, y):
    """
    Compute the gradient of the multiclass softmax cross-entropy
    with respect to its input variables, given only the output
    of the softmax function

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the softmax function
    y: ndarray(N)
        Target values
    
    Returns
    -------
    ndarray(N):
        Derivative of multiclass softmax cross-entropy
    """
    return y_est-y

def squared_loss(y_est, y):
    """
    Compute the squared loss with respect to its
    input variables, given the output of the layer before it

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the layer before the squared loss
    y: ndarray(N)
        Target of the output of the layer before
    
    Returns
    -------
    float:
        Squared loss
    """
    return np.sum((y-y_est)**2)

def squared_loss_deriv(y_est, y):
    """
    Compute the gradient of the squared loss with respect to its
    input variables, given the output of the layer before it

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the layer before the squared loss
    y: ndarray(N)
        Target of the output of the layer before
    
    Returns
    -------
    ndarray(N):
        Derivative of the squared loss
    """
    return 2*(y_est-y)

def abs_loss(y_est, y):
    """
    Compute the absolute value loss with respect to its
    input variables, given the output of the layer before it

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the layer before the squared loss
    y: ndarray(N)
        Target of the output of the layer before
    
    Returns
    -------
    float:
        Absolute value loss
    """
    return np.sum(np.abs(y-y_est))

def abs_loss_deriv(y_est, y):
    """
    Compute the gradient of the absolute value loss with respect to its
    input variables, given the output of the layer before it

    Parameters
    ----------
    y_est: ndarray(N)
        Output of the layer before the squared loss
    y: ndarray(N)
        Target of the output of the layer before
    
    Returns
    -------
    ndarray(N):
        Derivative of the squared loss
    """
    return np.sign(y_est-y)