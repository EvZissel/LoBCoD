%A Demo script for the image inpainting using LoBCoD algorithm


addpath mexfiles;
addpath image_helpers;
addpath('vlfeat/toolbox');
addpath('utilities');
addpath(genpath('spams-matlab'));
vl_setup();


imgs_path = 'datasets\single';
n = 8; %patch size
m = 81; %number of filters
[I,noisyI,M,noisyImean,lmnI] = Create_Zearo_Mask(imgs_path,n);
sz = size(I{1});
I{1} = I{1}(50:200,1:200);
noisyI{1} = noisyI{1}(50:200,1:200);
M{1} = M{1}(50:200,1:200);
noisyImean{1} = noisyImean{1}(50:200,1:200);
lmnI{1} = lmnI{1}(50:200,1:200);

D_initial = load('D_initial.mat');
D = D_initial.D_LoBCoD;

params = [];
params.Y_Original = I;
params.Y_noisy = noisyI;
params.lmnI = lmnI;
params.noisyImean = noisyImean;
params.M = M;
params.lambda = 0.01; 0.1;
params.MAXITER = 500; %default maximun number of epochs
params.D = D;
params.Train_on = true(1);

%% Run LoBCoD
[cleanI_inpainting,objective_inpainting,avgpsnr_inpainting,sparsity_inpainting,...
    totTime_inpainting,alpha_inpainting,D_inpainting] = Inpainting_LoBCoD(params);


%% Plot

figure(1); 
subplot(1,3,1);
plot(totTime_inpainting,objective_inpainting,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Objective','fontsize',10)
xlim([0 5500])
legend('LoBCoD inpainting')
grid on

subplot(1,3,2);
plot(totTime_inpainting,avgpsnr_inpainting,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Average PSNR','fontsize',10)
xlim([0 5500])
legend('LoBCoD')
grid on

subplot(1,3,3);
plot(totTime_inpainting,sparsity_inpainting,'.-b');
xlabel('Time [Seconds]','fontsize',10)
ylabel('Sparsity','fontsize',10)
xlim([0 5500])
legend('LoBCoD')
grid on

figure(2); 
showDictionary(D_inpainting);
title('The trained dictionary on the corrupted images');


figure(3);
imagesc(I{1}+ lmnI{1}); colormap gray
title('Original image')

figure(4);
imagesc(noisyI{1}+ M{1}.*noisyImean{1}); colormap gray
title('Corrupted image')

figure(5);
imagesc(cleanI_inpainting{1}+ noisyImean{1}); colormap gray
title(['Clean image PSNR = ',num2str(avgpsnr_inpainting(end)),'dB'])
