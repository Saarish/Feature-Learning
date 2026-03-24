% This funtion tests the performance of the SAE on 3 input audio files from tool/
% using learnt weights W1, b1, W2 and b2:
clear;
clc;
addpath tool/
load W1b1L1;
load W1b1L2;
% Initialising values for CQT of the input files:
fs = 16000;
bins_per_octave = 12;
fmax = fs/5;     %center frequency of the highest frequency bin = 3200 
fmin = fmax/64;  %lower boundary for CQT (lowest frequency bin will be immediately above this): fmax/<power of two> = 250
L_syllable=0.2/(1/fs);  % The syllable size is 3200
L_hop=10;
L_win=20;

% Extract features from the ground truth recordings:
gtFiles=dir('test/*.wav');
L_gt=length(gtFiles); % L_gt=3
patchInAllFiles=cell(L_gt,1); % each cell represents all the patches in one .wav file
numOfPatchInOneFile=zeros(L_gt,1); % each element represents the number of patches in one .wav file

for i=1:L_gt
    wav_gt=audioread(fullfile('test',gtFiles(i).name));
    Xcqt_gt=cqt(wav_gt,fmin,fmax,bins_per_octave,fs,'atomHopFactor',0.25*20); % constant Q transform of each .wav file
    intCQT_gt=getCQT(Xcqt_gt,'all','all'); % CQT output is a matrix, the y-axis dimensionality is 72
    len=size(intCQT_gt,2); % number of frames of the CQT spectrogram
    num=floor((len-L_hop)/L_hop); % number of patches need to be seperated
    cellpatch=cell(num,1);
    numOfPatchInOneFile(i)=num; % number of patches in one .wav file stored
    
    for k=1:num % reshape the patches into long vectors of 1*1440
        intCQT_patch=intCQT_gt(:,(k-1)*L_hop+1:(k-1)*L_hop+L_win);
        intCQT_patch=reshape(intCQT_patch,[72*L_win,1]);
        intCQT_patch=intCQT_patch';
        cellpatch(k)={intCQT_patch};
    end
    
    patchInOneFile=cell2mat(cellpatch);
    patchInOneFile=round(255*patchInOneFile/max(max(patchInOneFile))); % normalization of the patches in one .wav file
    
    % Silence removal by rms detection
    cout=0;
    for l=1:size(patchInOneFile,1)
        if rms(rms(patchInOneFile(l,:)))>=0.01
            cout=cout+1;
        end
    end
    numPatchAftRmv=cout;
    patchInOneFile_silence_rmv=cell(numPatchAftRmv,1);
    
    cout=0;
    for l=1:size(patchInOneFile,1)
        if rms(rms(patchInOneFile(l,:)))>=0.01
            cout=cout+1;
            patchInOneFile_silence_rmv(cout)={patchInOneFile(l,:)};
        end
    end
    patchInAllFiles(i)={cell2mat(patchInOneFile_silence_rmv)};
end
%--------------------------------------------------------------------------
% Having stored the variables W1b1L1 ad W1B1L2 in the workspace and calling
% them at the start of this function, we first separate the individual
% weights and biases for the 2 layers as W1,b1 and W2,b2 respectively:

W1 = W1b1L1(:,(1:length(W1b1L1)-1));
b1 = W1b1L1(:,length(W1b1L1));

W2 = W1b1L2(:,(1:length(W1b1L2)-1));
b2 = W1b1L2(:,length(W1b1L2));

% Initialising the test vector, a 100d mean vector representation of the 3
% input files:
test_vector = zeros(100,3);

% Now for each input out of the 3, I first extract a test_input vector from
% patchInAllFiles{}, calculate the intermediate input to the first hidden layer
% as a sigmoid function of weights W1,b1 and the test_input.
% This reduces the lengths to 500 from 1440 for all patches.
% Further, I calculate the final output to the second hidden layer using
% a sigmoid function of weights W2,b2 and the intermedeate_input, 
% which reduces the representation to 100d. Next, a mean over all patches would give the 
% necessary 100d representation of that particular input which can be stored in test_vector. 
% This is repeated for the 3 inputs:
for i =1:3
test_input = patchInAllFiles{i,1};
inter_input = sigmoid(W1*test_input' + repmat(b1,1,size(test_input,1)));
test_output = sigmoid(W2*inter_input + repmat(b2,1,size(test_input,1)));
test_vector(:,i)  = mean(test_output,2);
end

%--------------------------------------------------------------------------

fprintf('Distance between Marimba imitation and Marimba instrument: %f\n\n',...
        sum(abs(test_vector(:,1)-test_vector(:,2)).^2));

fprintf('Distance between Marimba imitation and Thaigong instrument: %f\n',...
        sum(abs(test_vector(:,1)-test_vector(:,3)).^2));
