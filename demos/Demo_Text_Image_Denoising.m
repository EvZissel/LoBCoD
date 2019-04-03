% Salt and Pepper Text Image Denoising Demo script.
% This demonstration was inspired by the implementation of:
%   "E. Plaut, R. Giryes, "A Greedy Approach to Convolutional Sparse Coding", 2018

addpath('functions')
addpath mexfiles;
addpath image_helpers;
addpath('vlfeat/toolbox');
addpath('utilities');
addpath(genpath('spams-matlab'));
vl_setup();

%constants
n = 11; % patch size
m = 100; % number of filters
TrainOn = 0; % If 1, train new dictionaries. If 0, load pre-trained dictionaries.
train_size = 16;

%load data
imgs_path = 'datasets\text_images';
I = Load_Text_Images(imgs_path);
x_train = cell(train_size,1);
x_test = cell(train_size,1);
for i=1:train_size
    x_train{i,1} = I{i};
end
for i= 1:(length(I)-train_size)
    x_test{i,1} = I{train_size+i};
end


%% Invert all images
x_train_inverted=cell(size(x_train));
x_test_inverted=cell(size(x_test));

for i=1:size(x_train,1)
    x_train_inverted{i,1}=1-x_train{i,1};
end
for i=1:size(x_test,1)
    x_test_inverted{i,1}=1-x_test{i,1};
end

%% Add salt and pepper noise
v=0.1; % noise probability including both salt and pepper
x_test_noisy=cell(size(x_test));
for i=1:size(x_test,1)
    noise = rand(size(x_test{i,1}))*2-1; 
    x_test_noisy{i,1} = x_test_inverted{i,1};
    x_test_noisy{i,1}(noise>1-v)=1;
    x_test_noisy{i,1}(noise<-1+v)=0;
end

%% Train a dictionary on noisy images
%  Note: For training on clean images, replace 'x_test_noisy' with 'x_train'
lambda_train = 0.1;
if (TrainOn)
    x_test_noisy_mean = cell(size(x_test_noisy));
    for i=1:size(x_train,1)
        x_test_noisy_mean{i,1} = x_test_noisy{i,1}-mean(x_test_noisy{i,1}(:));
    end
    params = [];
    params.Ytrain = x_test_noisy_mean;
    params.lambda = lambda_train;
    params.MAXITER = 50; 
    params.D = normc(randn(n^2,m));
    params.Train_on = true(1);
    
    [~,~,~,~,~,~,D_noisy] = LoBCoD(params);
    save('D_noisy.mat','D_noisy');
else
    load('D_noisy.mat')
end

%% Remove noisy atoms
th=0.528;
D_noisy = D_noisy(:,sum(abs(D_noisy-repmat(mode(D_noisy),[size(D_noisy,1),1]))>th)==0);

%% Crop centers of test images
for i=1:size(x_test,1)
    x_test{i,1} = x_test{i,1}(221:460,91:330);
    x_test_noisy{i,1} = x_test_noisy{i,1}(221-n:460+n,91-n:330+n);
end

%% Denoising using the dictionary trained on the noisy test set.
%  Note: For denoising using the dictionary trained on the clean training set
%  change the dictionary below.

x_denoised = cell(size(x_test));
max_iter = 1;
params = [];
lambda =  0.1;
params.MAXITER = 100; 
params.Train_on = false(1);
D = D_noisy;
for i=1:size(x_test,1)
    
    x_test_noisy_mean = mean(x_test_noisy{i,1}(:));
    x_test_noisy_i = x_test_noisy{i,1}-x_test_noisy_mean;
    x_text = cell(1);
    x_text{1}=0;
    sz = size(x_test_noisy_i);
    I_noise = cell(1);

    for iter=1:max_iter

        I_noise{1} = x_test_noisy_i-x_text{1};

        params.Ytrain = I_noise;
        params.lambda =  5*lambda;
        params.D = [1;zeros(120,1)];
        [~,~,~,~,~,alpha,~] = LoBCoD(params);
        [a_noise,~] = create_feature_maps(alpha,n,1,sz,[1;zeros(120,1)]);     
        a_noise{1} = hard_threshold(a_noise{1},0.5);

        [~,x_noise] = extract_feature_maps(a_noise,n,size([1;zeros(120,1)],2),sz,[1;zeros(120,1)]);
        x_noise{1} = x_noise{1}-mean(x_noise{1}(:));
        
        % represent the text image
        x_text{1} = x_test_noisy_i-x_noise{1};
        params.Ytrain = x_text;
        params.lambda =  lambda;
        params.D = [D [1; zeros(120,1)]];
        [~,~,~,~,~,alpha_text,~] = LoBCoD(params);
        [a_text,~] = create_feature_maps(alpha_text,n,size([D [1; zeros(120,1)]],2),sz,[D [1; zeros(120,1)]]); 
        [~,x_text] = extract_feature_maps(a_text,n,size([D [0; zeros(120,1)]],2),sz,[D [0; zeros(120,1)]]);
    end
    x_text{1} = x_text{1}+x_test_noisy_mean;
    
    x_denoised{i,1} = 1-x_text{1};
    x_denoised{i,1} = x_denoised{i,1}(n+1:end-n,n+1:end-n);
    x_test_noisy{i,1} = x_test_noisy{i,1}(n+1:end-n,n+1:end-n);
end

%% Plots

sum_denoised_psnr = 0;
for i=1:size(x_test,1)
   sum_denoised_psnr = sum_denoised_psnr+psnr(x_denoised{i,1},x_test{i,1});
end

% Average PSNR on test set
fprintf('Average PSNR = %.3f\n',sum_denoised_psnr/(size(x_test,1)));

figure;
subplot(1,3,1); imshow(x_test{1,1}); title('original','fontsize',14);
subplot(1,3,2); imshow(1-x_test_noisy{1,1}); title('noisy','fontsize',14);
subplot(1,3,3); imshow(x_denoised{1,1}); title('denoised','fontsize',14);


