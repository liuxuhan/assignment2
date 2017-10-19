from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x >= 0) * dout
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.binomial(1, p, size=x.shape)
        out = x * mask /(1-p)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = np.random.binomial(1, p, size=x.shape)
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        p = dropout_param['p']
        dx = dout /(1-p) * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def im2col(HH,WW,x,p,stride):
    N,C,XH,XW = x.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_indices(x.shape, HH, WW, p, stride)
    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(HH * WW * C, -1)
    return cols
    #return X

def col2im(dx_col, x_shape, HH, WW, padding, stride):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2*padding, W + 2*padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=dx_col.dtype)
    k, i, j = get_indices(x_shape, HH, WW, padding, stride)
    dx_col_reshaped = dx_col.reshape(C * HH * WW, -1, N)
    dx_col_reshaped = dx_col_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), dx_col_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
    
def get_indices(x_shape, HH, WW, padding=1, stride=1):
    N, C, H, W = x_shape
    Oh = int((H + 2 * padding - WW) / stride + 1)
    Ow = int((W + 2 * padding - WW) / stride + 1)
    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(Oh), Ow)
    j0 = np.tile(np.arange(WW), HH * C)
    j1 = stride * np.tile(np.arange(Ow), Oh)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def conv_forward(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction.
         We will use the same definition in lecture notes 3b, slide 13 (ie. same padding on both sides).

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + pad - HH) / stride
      W' = 1 + (W + pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N,C,XH,XW = x.shape
    F,C,HH,WW = w.shape
    stride = conv_param['stride']
    padding = conv_param['pad']
    Oh = int(1 + ( XH + padding - HH)/stride)
    Ow = int(1 + ( XW + padding - WW)/stride)
    W = w.reshape(F, -1)
    B = np.tile(b, (N*Oh*Ow,1)).T
    X_col = im2col(HH,WW,x,int(padding/2),stride)
    out = W@X_col+B
    
    out = out.reshape(F, Oh, Ow, N).transpose(3,0,1,2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    #print(X)
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param= cache
    F, C, HH, WW = w.shape
    N,C,XH,XW = x.shape
    stride = conv_param['stride']
    padding = conv_param['pad']
    Oh = int(1 + ( XH + padding - HH)/stride)
    Ow = int(1 + ( XW + padding - WW)/stride)
    
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
    db = np.sum(dout_reshaped,axis=1)
    
    x_col = im2col(HH,WW,x,int(padding/2),stride)
    dw = dout_reshaped @ x_col.T
    dw = dw.reshape(w.shape)

    w_reshape = w.reshape(F, -1)
    dx_col = w_reshape.T @ dout_reshaped
    dx = col2im(dx_col, x.shape, HH, WW, int(padding/2), stride)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def _pool_forward(X, pool_func, pool_param):
    n, d, h, w = X.shape
    ph= pool_param['pool_height']
    pw= pool_param['pool_width']
    stride = pool_param['stride']
    h_out = int((h - ph) / stride + 1)
    w_out = int((w - pw) / stride + 1)
  
    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col(ph, pw,X_reshaped, 0, stride) # shape ( size*size,n*d*h_out*w_out)
    out = pool_func(X_col)
    out = out.reshape(h_out, w_out, n, d).transpose(2, 3, 0, 1)
   
    cache = (X, pool_param)
    return out, cache

def max_pool_forward(x, pool_param):
    """
    Forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    def maxpool(X_col):
        out = np.max(X_col,axis=0)
        return out

    return _pool_forward(x, maxpool, pool_param)


def max_pool_backward(dout, cache):
    """
    Backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    X, pool_param = cache
    stride = pool_param['stride']
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    
    n, d, w, h = X.shape
    
    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col(ph, pw,X_reshaped, 0, stride)
    max_idx = np.argmax(X_col, axis=0)
    
    dx_col = np.zeros_like(X_col)
    dx = np.zeros_like(X_col)
    
    dout_col = dout.transpose(2, 3, 0, 1).ravel()
    
    dx_col[max_idx, range(max_idx.size)] = dout_col
    dx = col2im(dx_col,X_reshaped.shape , ph, pw, 0, stride)
    dx = dx.reshape(X.shape)
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def l2_reg(W, reg=0.0):
    return .5 * reg * np.sum(W * W)

def dl2_reg(W, reg=0.0):
    return reg * W
