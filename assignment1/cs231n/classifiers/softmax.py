import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
  that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = len(X)
  num_feature = W.shape[1]
  output = X.dot(W)
  output_exp = np.exp(output)
  
  for i in range(num_train):
      s = np.sum(output_exp[i])
      p = output_exp[i]/s
      
      loss -= np.log(p[y[i]])
      for j in range(num_feature):
          dW[:,j] += X[i]*p[j] - (j==y[i])*X[i]
#      dW += X[i][:,np.newaxis].dot(p[np.newaxis,:])
#      print (X[i][:,np.newaxis].dot(p[np.newaxis,:]).shape, dW.shape)
#      assert False
#      dW[:,y[i]] -= X[i][y[i]]
  loss /= num_train
  loss += reg * np.sum(W * W)
  assert loss > 0
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = len(X)
  num_feature = W.shape[1]
  output = X.dot(W)
  output_exp = np.exp(output)
  p = output_exp/np.sum(output_exp, axis = 1)[:,np.newaxis]
  loss = -np.sum(np.log(p[np.arange(num_train), y]))/num_train + reg * np.sum(W*W)
#  X_sum = np.sum(X, axis = 0)
  pick_arr = np.zeros((num_train, num_feature))
  pick_arr[np.arange(num_train), y] = 1
  dW += X.transpose().dot(p) - X.transpose().dot(pick_arr)
  
#  dW -= X.transpose().dot(pick_arr)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

