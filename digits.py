import numpy as np
import matplotlib.pyplot as plt
from mlp import *
from losses import *
from layers import *

def get_digits(foldername):
    """
    Load all of the digits 0-9 in from some folder

    Parameters
    ----------
    foldername: string
        Path to folder containing 0.png, 1.png, ..., 9.png
    
    Returns 
    -------
        X: ndarray(n_examples, 28*28)
            Array of digits
        y: ndarray(n_examples, dtype=int)
            Number of each digit
    """
    res = 28
    X = []
    Y = []
    for num in range(10):
        I = plt.imread("{}/{}.png".format(foldername, num))
        row = 0
        col = 0
        while row < I.shape[0]:
            col = 0
            while col < I.shape[1]:
                img = I[row:row+res, col:col+res]
                if np.sum(img) > 0:
                    X.append(img.flatten())
                    Y.append(num)
                col += res
            row += res
    return np.array(X), np.array(Y, dtype=int)

def scatter_digits(model, X, Y, n_scatter=1000):
    """
    Scatter a subset of digits in their latent representation
    
    Parameters
    ----------
    X: ndarray(N, 28*28)
        Digits
    Y: ndarray(N)
        Digit labels
    model: MLP object
        Autoencoder model.  Should take in 28*28 dimensions and output 28*28 dimensions,
        with a 2D layer in the middle named "latent"
    n_scatter: int
        Number of example digits to scatter
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    ax = plt.gca()
    encoded = []
    # Convert a grayscale digit to one with a background color chosen from
    # the tab10 colorcycle to indicate its class
    c = plt.get_cmap("tab10")
    jump = X.shape[0]//n_scatter
    for k in range(n_scatter):
        k = k*jump
        x = X[k, :]
        label = Y[k]
        x, y = model.forward(x, end="latent")
        encoded.append([x, y])
        img = np.reshape(X[k, :], (28, 28))
        C = c([label]).flatten()[0:3]
        img_disp = np.zeros((28, 28, 4))
        img_disp[:, :, 0:3] = (1-img)[:, :, None]*C[None, None, :]
        img_disp[:, :, 3] = 1-img
        img = OffsetImage(img_disp, zoom=0.7)
        ab = AnnotationBbox(img, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
    encoded = np.array(encoded)
    ax.update_datalim(encoded)
    ax.autoscale()

def plot_digits_dimreduced_examples(X, Y, model, n_examples=20):
    """
    Plot examples of encoded digits, as well as a scatter of some digits
    in their latent representation
    
    Parameters
    ----------
    X: ndarray(N, 28*28)
        Digits
    Y: ndarray(N)
        Digit labels
    model: MLP object
        Autoencoder model.  Should take in 28*28 dimensions and output 28*28 dimensions,
        with a 2D layer in the middle named "latent"
    n_examples: int
        Number of example encodings to show
    """
    ## Step 1: Plot examples of encodings
    jump = X.shape[0]//n_examples
    for k in range(n_examples):
        tidx = k*jump
        x = X[tidx, :]
        plt.subplot(n_examples, n_examples, k+1)
        plt.imshow(np.reshape(x, (28, 28)), vmin=0, vmax=1, cmap='gray')
        plt.axis("off")
        plt.subplot(n_examples, n_examples, n_examples+k+1)
        y = model.forward(x)
        y = np.reshape(y, (28, 28))
        plt.imshow(np.reshape(y, (28, 28)), vmin=0, vmax=1, cmap='gray')
        plt.axis("off")

    ## Step 2: Do a scatterplot of a subset of the digits in their latent space
    plt.subplot2grid((n_examples, n_examples), (2, 0), colspan=n_examples, rowspan=n_examples-2)
    scatter_digits(model, X, Y)