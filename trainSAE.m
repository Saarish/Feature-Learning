function [W1, b1, W2, b2] = trainSAE(visibleSize,hiddenSizeL1,hiddenSizeL2,lambda,input_layer)
% Function to train a stacked auto-encoder (SAE) with 2 hidden layers
%
% Input
%  - visibleSize     : size of input layer
%  - hiddenSizeL1 : size of 1st hidden layer 
%  - hiddenSizeL2 : size of 2nd hidden layer
%  - lambda         : regularization parameter
%  - X                : input data matrix
% 
% Output
%  - W1             : trained weights to the first hidden layer
%  - b1              : trained biases to the first hidden layer
%  - W2             : trained weights to the second hidden layer
%  - b2              : trained biases to the second hidden layer

% Author: Saarish Kareer
% Created: 10/24/17
% Last modified: 10/24/17

%-----------------------------------------------------------------------
% Training the first auto-encoder
% Initialize parameters:
theta_1 = initializeParameters(hiddenSizeL1, visibleSize);

% Using the lbfgsFunc advanced optimisation function that uses minFunc, I
% have calculated and displayed the optimal weights and minimised cost function for the first
% autoencoder. The Input is the input_layer from a Vocal Imitation dataset
% along with corresponding lengths and regularisation paramters, and this in turn goes into the
% autoencoderCost.m to calculate the gradients and cost at each iteration
% of the optimization process. Finally the weights and biases can be split
% up into respective dimensions using reshape.

[optimumTheta_1, cost_1] = lbfgsFunc(visibleSize, hiddenSizeL1, lambda, input_layer, theta_1) ; 

W1 = reshape(optimumTheta_1(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
b1 = optimumTheta_1(2*hiddenSizeL1*visibleSize+1:2*hiddenSizeL1*visibleSize+hiddenSizeL1);
fprintf('Minimised Cost from the 1st auto-encoder is %s\n',cost_1);

%-------------

% Training the second auto-encoder
% Initialize parameters:
theta_2 = initializeParameters(hiddenSizeL2, hiddenSizeL1);

% Using the lbfgsFunc advanced optimisation function that uses minFunc, I
% have calculated and displayed the optimal weights and minimised cost
% function for the second autoencoder. The Input is the hidden_layer
% calculated from the optimised weights and biases of the first AE and the
% input_layer as follows:
hidden_layer = sigmoid([b1 W1]*[ones(1,length(input_layer));input_layer]);

% Also given are corresponding layer lengths and regularisation paramters, and this in turn goes into the
% autoencoderCost.m to calculate the gradients and cost at each iteration
% of the optimization process. Finally the weights W2 and biases b2 can be split
% up into respective dimensions using reshape, and the minimised final cost is
% displayed:
[optimumTheta_2, cost_2] = lbfgsFunc(hiddenSizeL1, hiddenSizeL2, lambda, hidden_layer,theta_2);

W2 = reshape(optimumTheta_2(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
b2 = optimumTheta_2(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
fprintf('Minimised Cost from the 2nd auto-encoder is %s\n',cost_2);

%------------