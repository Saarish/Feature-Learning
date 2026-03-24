% This function reads in the vocal imitations and trains a 2 layer SAE for
% feature learning:
addpath tool/
% wav file into patches
fs = 16000;
bins_per_octave = 12;
fmax = fs/5;     %center frequency of the highest frequency bin 
fmin = fmax/64; %lower boundary for CQT (lowest frequency bin will be immediately above this): fmax/<power of two> 
L_syllable=0.2/(1/fs);          % The syllable size is 3200
L_hop=420;                       % The hop size is 420
L_win=20;
Imitation_file=dir('vocal_imitations/*.wav');
L_Imitation=length(Imitation_file);
cellrecording=cell(L_Imitation,1);
numSum=0;

for i=1:1:L_Imitation
    wav_Imitation=audioread(fullfile('vocal_imitations',Imitation_file(i).name));
    len_wav=length(wav_Imitation);
    Xcqt = cqt(wav_Imitation,fmin,fmax,bins_per_octave,fs,'atomHopFactor',0.25*20);
    intCQT=getCQT(Xcqt,'all','all');
    len=size(intCQT,2);
    num=floor(len/10);
    numSum=numSum+num;
    cellpatch=cell(num,1);
    index=randi(len-L_win,num,1);
    for j=1:1:num
            cellpatch(j)={intCQT(:,index(j):index(j)+L_win-1)};
    end
    patch1=cell2mat(cellpatch);
    patch1=round(255*patch1/max(max(patch1)));
    fprintf('>');
    if mod(i,20)==0
        fprintf('\n');
    end
    cellrecording(i)={patch1};
end
recording=cell2mat(cellrecording);

% Remove silence patches
[M,N]=size(recording);
num=M/72;
count=0;
for i=1:num
    patch=recording((i-1)*72+1:i*72,:);
    if rms(rms(patch))>0.01
        count=count+1;
    end
end
numAftRemv=count;
cellpatch=cell(numAftRemv,1);

count=0;
for i =1:num
    patch=recording((i-1)*72+1:i*72,:);
    if rms(rms(patch))>0.01
        count=count+1;
        cellpatch(count)={patch};
    end
end

recordingAftRemv=cell2mat(cellpatch);
save('dataAftRemv.mat','recordingAftRemv');

% train the SAE
visibleSize=1440; % number of input neurons of the 1st AE 
hiddenSizeL1=500;   % number of hidden neurons of the 1st AE 
hiddenSizeL2=100;   % number of hidden neurons of the 2nd AE
% sparsityParam=0.01; % desired average activation of the hidden units.
%                      % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p"). 
lambda=0.0001;  % weight decay parameter  
% beta=3; % weight of sparsity penalty term
patches = sampleIMAGES_AftRemv;

% This line of code, runs the training on my SAE with the paramters and lengths given
% above & the input_layer (patches) defined from the imitation data above.
% It outputs W1,b1 - The trained and optimised weights for the 1st hidden
% Layer in the autoencoder 1, and W2,b2 - The trained and optimised weights
% for the 1st hidden Layer in the autoencoder 2:

[W1, b1, W2, b2] = trainSAE(visibleSize,hiddenSizeL1,hiddenSizeL2,lambda,patches);

