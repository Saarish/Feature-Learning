function [Cost,grad] = autoencoderCost(theta, visibleSize, hiddenSize, lambda, patches)
% This function calculates the overall cost of an auto-encoder on all input
% data, and the partial derivatives of the cost w.r.t all weights.
%
% Input
%  - theta          : all weights arranged as a vector. Its length should be
%                       2*visibleSize*hiddenSize (for W1 and W2) +
%                       hiddenSize (for b1) + visibleSize (for b2)
%  - visibleSize    : input layer size
%  - hiddenSize   : hidden layer size
%  - lambda        : parameter for the regularization term
%  - X               : the input data matrix. Each column is an example
% 
% Output
%  - cost           : the overall error cost J(W,b) that we want to
%                       minimize. A scalar
%  - grad           : the gradients of the cost w.r.t. to all weights. A
%                       vector of the same size as theta.
%
% Author: Saarish Kareer
% Created: 10/24/17
% Last modified: 10/24/17

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% ----------------------------------------------------------------------
% First, defining the number of training examples as m:
m = length(patches);

% Using my forwardActivation.m function to calculate a2 (first Hidden
% Layer) and a3 (second Hidden Layer ~ output layer) given the randomly initialized weights and biases
% from above and the feedforward strategy:
[a2, a3] = forwardActivation(W1, W2, b1, b2, patches);

% The Backpropagation algorithm starts here, I first calculate the delta
% errors associated with each layer, a3 and a2, and call them delta3 and
% delta2 respectively. Then I find an intermediate array Del_W and Del_B for the Gradients
% corresponding to the weights and biases:

delta3 =  -(patches - a3).*(a3.*(1-a3)) ; 
delta2 = (W2'*delta3).*((a2).*(1-a2));

Del_W1  = delta2*patches';
Del_W2  = delta3*a2';

Del_b1  =  delta2;
Del_b2  =  delta3;

% The cost at every iteration and display next along with regularization applied for the weights: 

Cost  = (1/m)*(sum(sum(((patches - a3).^2)/2))) + (lambda/2)*( sum(sum(W1.^2)) + (sum(sum(W2.^2))));
fprintf('The cost at this iteration is %s\n',Cost);

% Finally, using the number of training examples and nodes we calculate the
% weights gradient matrices and unroll them into a vector suitable for the optimization function: 

W1grad = Del_W1/m + lambda*W1;
W2grad = Del_W2/m + lambda*W2;
b1grad = sum(Del_b1,2)/m ;
b2grad = sum(Del_b2,2)/m ;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
%--------------------------------------------------------------------------
end