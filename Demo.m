%A Demo script for the LoBCoD algorithm

% This demo shows how to applay the LoBCoD algorithm to train the
% CSC dictionary

addpath mexfiles;
addpath image_helpers;
addpath('vlfeat/toolbox');
addpath('utilities');
addpath(genpath('spams-matlab'));
vl_setup();


imgs_path = 'datasets\fruit_100_100';
n = 8; %patch size
m = 81; %number of filters
I = Create_Zearo_Mean_Images(imgs_path,n);

D = randn(n^2,m);
Dn = diag(1./sqrt(diag(D'*D)));
D = D*Dn;

params = [];
params.Ytrain = I;
params.lambda = 20;
params.MAXITER = 500; %default maximun number of iterations
params.D = D;
params.Train_on = true(1);

%% Run LoBCoD
[cleanI,objective_LoBCoD,avgpsnr_LoBCoD,sparsity_LoBCoD,totTime_LoBCoD,alpha,D_LoBCoD] = LoBCoD(params);


%% Plot

figure(1); 
subplot(1,3,1);
plot(totTime_LoBCoD,objective_LoBCoD,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Objective','fontsize',10)
legend('LoBCoD')
grid on

subplot(1,3,2);
plot(totTime_LoBCoD,avgpsnr_LoBCoD,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Average PSNR','fontsize',10)
legend('LoBCoD')
grid on

subplot(1,3,3);
plot(totTime_LoBCoD,sparsity_LoBCoD,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Sparsity','fontsize',10)
legend('LoBCoD')
grid on

figure(2); 
showDictionary(D_LoBCoD);
title('Trained dictionary LoBCoD');
