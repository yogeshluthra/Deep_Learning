import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def ReLU(self, X):
    """applies non-linearity on X"""
    Xp=np.copy(X)
    Xp[0.0>Xp]=0.0
    return Xp

  def softmax(self, X):
    Xp =  np.e**X/np.sum(np.e**X, axis=1).reshape(-1,1)
    return Xp

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    num_classes=W2.shape[1]

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    g1=X.dot(W1)+b1
    a1=self.ReLU(g1)

    g2=a1.dot(W2)+b2
    scores=g2 # score is un-normalized log probability
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    a2 = self.softmax(g2) # convert log probability to probability

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    L=-1.*np.log(a2)
    L=L[np.arange(N), y]
    loss = 1./N * np.sum(L) + 0.5*reg*(np.sum(W1**2)+np.sum(W2**2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # - initialize
    g=np.zeros(shape=(N, num_classes), dtype=float)

    # - D_a2{loss} = D_a2{L} * D_L{loss} = -1/a2[np.arange(N), y] * 1/N
    g[np.arange(N), y] = (-1./a2[np.arange(N), y]) * (1./N)
    # print 'D_a2{loss} = D_a2{L} * D_L{loss} = -1/a2[np.arange(N), y] * 1/N'
    # print g[:5]
    # print

    # - D_g2{loss} = D_g2{a2} * D_a2{loss}  (most tricky !!!)
    g_g2 = np.zeros(shape=(N, num_classes), dtype=float)
    # every g_g2[i] row vector needs to be (diag(a2[i]) - a2[i].a2[i]').D_a2{loss}[i] ; where a2[i] is a row vector (that is specific to an example)
    for i in range(N):
      g_g2[i] = ( np.diag(a2[i]) - a2[i].reshape(-1,1).dot(a2[i].reshape(1,-1)) ). \
                                          dot(g[i])
    g=g_g2
    # g = a2*(1.-a2) * g
    # print "D_g2{loss} = D_g2{a2} * D_a2{loss} = a2*(1-a2) * g"
    # print g[:5]
    # print

    # - D_W2{loss} = D_W2{g2} . D_g2{loss} = D_W2{g2} . ( D_g2{a2} * D_a2{loss} )= a1' . g
    # - D_b2{loss} = D_g2{loss} (but summed across all examples in this minimatch)
    grads['W2'] = a1.T.dot(g) + 1.*reg*W2
    grads['b2'] = np.sum(g, axis=0)

    # - D_a1{loss} = D_a1{g2} . D_g2{loss} = g . W2'
    g=g.dot(W2.T)
    # print "D_a1{loss} = D_a1{g2} . D_g2{loss} = g . W2'"
    # print g[:5]
    # print

    # - D_g1{loss} = D_g1{a1} * D_a1{loss} = 1_g1>0 * D_a1{loss}
    g[0.0>g1]=0.0 # this ReLU is not leaky, so clips at 0.0
    # print "D_g1{loss} = D_g1{a1} * D_a1{loss} = 1_g1>0 * D_a1{loss}"
    # print g[:5]
    # print

    # - D_W1{loss} = D_W1{g1} . D_g1{loss} = X' . g
    # - D_b1{loss} = G_g1{loss} (but summed across all examples in this mini-batch
    grads['W1'] = X.T.dot(g) + reg*W1
    grads['b1'] = np.sum(g, axis=0)


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_indices = np.random.choice(np.arange(num_train), size=batch_size, replace=False)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for param_name in self.params:
        self.params[param_name] += -learning_rate * grads[param_name]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


