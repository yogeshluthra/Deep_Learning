import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from collections import deque


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - [spatial batch norm] - relu - 2x2 max pool - affine - [batch norm] - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, verbose=False, use_batchnorm=False, use_running_stat=True):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.verbose=verbose
    self.use_batchnorm=use_batchnorm
    self.NetworkFlow = [] # stores tuples (params, forward_func, backward_func).
                          # TODO: Must make sure order of parameter passing between forward and backward pass is same.
                          # e.g. if forwardpass(x, w, b) ... then backwardpass must return dx, dw, db ...(note the order)
    self.paramsToRegularize = []
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train', 'use_running_stat': use_running_stat, 'layer': i + 1} for i in range(2)] # we know this network is only 3 layers (1 conv, 1 hidden, 1 affine)

    C, H, W=input_dim
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
    # conv layer (TODO: Assuming output of Conv layer is non-shrinking. So adding sufficient padding. Check this assumption?)
    F, HH, WW=num_filters, filter_size, filter_size
    self.params['W1']=np.random.normal(loc=0.0, scale=weight_scale, size=(F, C, HH, WW))
    self.params['b1']=np.zeros(F)
    stride=1
    pad=(H*(stride-1) + (HH - stride))/2 # TODO: In all of codes in this project, it is assumed the picture and filters are squares. This may not always be true!
    conv_params={'stride': 1, 'pad': pad}
    self.NetworkFlow.append((conv_forward_fast, ['W1', 'b1'], [conv_params], conv_backward_fast))
    self.paramsToRegularize.append('W1')

    # batch norm layer
    if self.use_batchnorm:
      self.params['gamma1']=np.ones(F)
      self.params['beta1']=np.zeros(F)
      self.NetworkFlow.append((spatial_batchnorm_forward, ['gamma1', 'beta1'], [self.bn_params[0]], spatial_batchnorm_backward))

    # relu
    self.NetworkFlow.append((relu_forward, [], [], relu_backward))

    # max pool
    pool_params={'pool_height': 2, 'pool_width': 2, 'stride': 2}
    self.NetworkFlow.append((max_pool_forward_fast, [], [pool_params], max_pool_backward_fast))
    inputDim_to_affine=F*H/2*W/2 # reduction in H and W after maxpool

    # affine
    self.params['W2']=np.random.normal(loc=0.0, scale=weight_scale, size=(inputDim_to_affine, hidden_dim))
    self.params['b2']=np.zeros(hidden_dim)
    self.NetworkFlow.append((affine_forward, ['W2', 'b2'], [], affine_backward))
    self.paramsToRegularize.append('W2')

    # batch norm layer
    if self.use_batchnorm:
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)
      self.NetworkFlow.append((batchnorm_forward, ['gamma2', 'beta2'], [self.bn_params[1]], batchnorm_backward))

    # relu
    self.NetworkFlow.append((relu_forward, [], [], relu_backward))

    # affine
    self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    self.NetworkFlow.append((affine_forward, ['W3', 'b3'], [], affine_backward))
    self.paramsToRegularize.append('W3')

    # endregion
    if self.verbose:
      print
      print "Network constructed is:"
      print '........'
      for item in self.NetworkFlow:
        print item
        print
      print

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    flowStack = deque()
    h = X  # input to first layer
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    for i, (forwardPass, paramKeys, extraParams, backwardPass) in enumerate(self.NetworkFlow):
      h, cache = forwardPass(h, *([self.params[key] for key in paramKeys]+extraParams))
      flowStack.append((backwardPass, paramKeys, cache))
    scores = h
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
    loss, dout = softmax_loss(scores, y)
    while len(flowStack) > 0:
      backwardPass, paramKeys, cache = flowStack.pop()
      layer_grads = backwardPass(dout, cache)
      dout = layer_grads[0]  # gradient for down-stream layer
      for i, key in enumerate(paramKeys):  # extract param gradients
        grads[key] = layer_grads[
          i + 1]  # TODO: Must make sure order of parameter passing between forward and backward pass is same.
        # e.g. if forwardpass(x, w, b) ... then backwardpass must return dx, dw, db ...(note the order)

    for paramToRegularize in self.paramsToRegularize:
      loss += 0.5 * self.reg * np.sum(self.params[paramToRegularize] ** 2)  # L2 norm
      grads[paramToRegularize] += self.reg * self.params[paramToRegularize]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
