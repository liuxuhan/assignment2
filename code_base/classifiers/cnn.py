from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(scale=weight_scale,size=(num_filters,C,filter_size,filter_size))
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(int(num_filters*H/2*W/2), hidden_dim))
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))

    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
    for k, v in self.params.items():
        self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    out, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) # convoluation relu max pool layer
    # Hidden Layer
    if self.use_dropout:
        out, cache_drop = dropout_forward(out, self.dropout_param)
    hidden_affine_out, hidden_affine_cache = affine_forward(out, W2, b2) # hidden layer
    relu_out,relu_cache = relu_forward(hidden_affine_out)                 # affine relu
    
    # if self.use_dropout:
    #     relu_out, cache_drop2 = dropout_forward(relu_out, self.dropout_param)
    output_affine_out, output_affine_cache = affine_forward(relu_out, W3, b3) # output layer
    scores = output_affine_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    if y is None:
        return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1']) 
         + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3']))
    
    output_affine_dx, output_affine_dw, output_affine_db = affine_backward(dscores, output_affine_cache)
    grads['W3'] = output_affine_dw + self.reg * self.params['W3']
    grads['b3'] = output_affine_db
     
    rule_dx = relu_backward(output_affine_dx, relu_cache)
    
    hidden_affine_dx, hidden_affine_dw, hidden_affine_db = affine_backward(rule_dx, hidden_affine_cache)
    grads['W2'] = hidden_affine_dw + self.reg * self.params['W2']
    grads['b2'] = hidden_affine_db
    
    if self.use_dropout:
        hidden_affine_dx = dropout_backward(hidden_affine_dx, cache_drop) 
        
        
    conv_dx, conv_dw, conv_db = conv_relu_pool_backward(hidden_affine_dx, cache)
    grads['W1'] = conv_dw + self.reg * self.params['W1']
    grads['b1'] = conv_db
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
pass
