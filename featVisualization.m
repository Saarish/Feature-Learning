function[] = featVisualization(W_vector)
% This function generates a visualization of the optimised weights of a
% particular Hidden Layer of the SAE. W can be passed in as a
% 500*1440 matrix, and we need to use the first 100 filters, representing
% the first 100 nodes of the hidden layers, for a visualization. This then
% is stored in V (100*1440), and for each of these 100 filters, a
% representation of patches (72*20) is obtained. This gives us 100 such
% patches which can be represented as a 10*10 Matrix and plotted using
% imagesc:

k=0;
V  = W_vector((1:100),(1:1440));
M= zeros(72,200);  % Initialising the matrix which stores all 100*1440 elements
for v = 0:10:90    
    j = 0;
    for i=1+v:10+v % The inner loop chooses the first 10 rows of V 
    
    sqr = reshape(V(i,:),72,20); % Reshaping into a 72*20 patch

    M(1+k:72+k,(1+j:20+j)) = sqr ; % Matrix M is filled up horizontally and is now 72*200 representing 10 filter
    j= j + 20;

    end 
   
    k = k + 72; % The outer loop caters to the next 10 filters ie (11-20,21-30 and so on) and 
                % correspondingly the matrix M is filled up vertically to
                % 720*200 which is the target representation.
    
end 
figure()
imagesc(M);
set(gca,'YDir','Normal');
xlabel('Time - Samples of 20 points each');
ylabel('Frequency - CQTs of 72 points each');
title('Visualization of the Weights of 100 filters in the first Hidden Layer of the SAE');






