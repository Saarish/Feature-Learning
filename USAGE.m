%% Stacked AutoEncoders for Vocal Imitation Detection - Saarish Kareer

% Usage:
% This script implements a Stacked Auto-Encoder for automatic feature learning from vocal 
% imitations. 

% Requirements: 
% Please add the entire folders 'minFunc', 'test', 'tool', 'vocal_imitations'
% and its subfolders into the Matlab (2016a Version) path. 

% Functions used: 
% trainSAE_VocalImitation.m, trainSAE.m, lbfgsFunc.m, minFunc.m,
% autoencoderCost.m, forwardActivation.m, sigmoid.m, featVisualization.m
% and test.m in a sequence for this implementation. 
clc
clearvars

% TRAINING : Extracting over 6000 vocal imitation patches (each a 72X20 Overlapping CQT Spectrograms) 
% from the given dataset to form an input_layer of dimensions 6000 X 1440. The function
% 'trainSAE_VocalImitation' contains trainSAE.m which is my
% function to train 2 auto-encoders, with the input_layer as input to the
% first and the hidden_layer being input to the second. This further uses my function
% autoencoderCost.m to calculate the trained and optimised weights for the first Layer of both
% the auto-encoders using advanced optimisation LBGSFunc. forwardActivation.m and
% autoencoderCost.m work in tandem for backpropogation to find the cost and weight gradients 
% at every iteration. Finally the optimised features from the first layers of both the encoders 
% are stored and returned as W1,B1 and W2,B2 respectively.

% The training process takes 35 minutes on a 3.3GHz i5 core and 16GB RAM Mac and 5 minutes on an 
% online GPU. The next line is commented as the model weights were trained already.

%--------- Please Uncomment this section for Re-training the Model---------
%            and storing the weights and biases for the SAEs

trainSAE_VocalImitation;
W1b1L1 = [W1 b1];
W1b1L2 = [W2 b2];
save('W1b1L1.mat','W1b1L1');
save('W1b1L2.mat','W1b1L2');

%--------------------------------------------------------------------------

% VISUALISATION : Using my function featVisualization.m, we visualize the weights of
% the first hidden Layer of the first autoencoder - W1. The first 100 (out of 500) of 
% the weights excluding the bias are used in this function, which represent filters for the first
% 100 nodes of the first hidden Layer and more essentially what these nodes learn during training.
% W1B1L1 are the pre-learned weights and biases from the 1st autoencoder saved in directory,
% we can use these, or the same variable from above if training is done again:
load W1b1L1;
featVisualization(W1b1L1);

% TESTING : The function testSAE_VocalImitation runs the SAE implementation on 3 new inputs
% (1 imitation and 2 instruments) that have the same patch length and forms a mean 100d vector 
% representation for each. It then compares the imitation vector with the instrument vectors and
% finds the corresponding Euiclidean distance, suggesting proximity of the
% imitation to its correct instrument. 

testSAE_VocalImitation;