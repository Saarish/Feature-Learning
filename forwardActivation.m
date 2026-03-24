function [a2, a3]=forwardActivation(W1, W2, b1, b2, X)
% Forward calculation of auto-encoder on a set (say k) of input examples
% Suppose the input and output layer each have n nodes and the hidden layer
% has m nodes.
%
% Input
%  - W1     : weights from input layer to hidden layer, (m*n) matrix
%  - W2     : weights from hidden layer to output layer, (n*m) matrix
%  - b1      : bias weights for the hidden layer, (m*1) vector
%  - b2      : bias weights for the output layer, (n*1) vector
%  - X       : input data, (n*k) matrix. Each column is one example. The number of
%               columns k is the number of examples. 
% Output
%  - a2      : output (activation) of the hidden layer, (m*k) matrix
%  - a3      : output (activation) of the output layer, (n*k) matrix
% 
% Author: Saarish Kareer
% Created: 10/24/17
% Last modified: 10/24/17

%------------------------------------------------------------------------
% The first hidden layer a2 can be formed using the vectorized approach
% equivalent to W1*Input_layer + b1 :
 a2 = sigmoid([b1 W1]*[ones(1,length(X));X]);
 
% The second hidden layer a3 can be formed using the vectorized approach
% equivalent to W2*Hidden_layer1 + b2:
 a3 = sigmoid([b2 W2]*[ones(1,length(a2));a2]);

%------------------------------------------------------------------------
end
