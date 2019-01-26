% A Demo script for the online execution of the LoBCoD algorithm.

% This demo makes use of 40 images uniformly sampled from the 'mirflickr'
% dataset for training the dictionary, and 5 different images for a testing
% set from the same source. For speeding up the computations, The images 
% are cropped to the size of 256x256 pixels, normalized, and mean substructed.  


addpath mexfiles;
addpath image_helpers;
addpath('vlfeat/toolbox');
addpath('utilities');
addpath(genpath('spams-matlab'));
vl_setup();


imgs_path = 'datasets\mirflickr_40samples_train';
imgs_path_test = 'datasets\mirflickr_test';
n = 8; % patch size
m = 81; % number of filters
[I,I_test,~,~] = Preparation(imgs_path,imgs_path_test,n);

D = randn(n^2,m);
Dn = diag(1./sqrt(diag(D'*D)));
D = D*Dn;

params = [];
params.Ytrain = I;
params.lambda =  0.1;
params.MAXITER = 201; % default maximum number of epochs
params.D = D;
params.eval_step = 200; % number of training images in the training phase (used before the evaluation stage).
params.Ytest = I_test;

%% Run LoBCoD
[cleanI_online,objective_LoBCoD_online,avgpsnr_LoBCoD_online,sparsity_LoBCoD_online,totTime_LoBCoD_online,...
    alpha_online,D_LoBCoD_online,objective_Test,time_Test,iteration_vec] = LoBCoD_online(params);


%% Plot

figure(1); 
subplot(1,3,1);
plot(totTime_LoBCoD_online,objective_LoBCoD_online,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Objective','fontsize',10)
xlim([0 5500])
legend('LoBCoD')
grid on

subplot(1,3,2);
plot(totTime_LoBCoD_online,avgpsnr_LoBCoD_online,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Average PSNR','fontsize',10)
xlim([0 5500])
legend('LoBCoD')
grid on

subplot(1,3,3);
plot(totTime_LoBCoD_online,sparsity_LoBCoD_online,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Sparsity','fontsize',10)
xlim([0 5500])
legend('LoBCoD')
grid on

figure(2); 
showDictionary(D_LoBCoD_online);
title('Trained dictionary online LoBCoD');

figure(3);
plot(time_Test,objective_Test,'b-x','lineWidth',1);
xlabel('Time [Seconds]','fontsize',10)
ylabel('Objective on test set','fontsize',10)
title('The Objective value on the test set');
legend('LoBCoD')
grid on