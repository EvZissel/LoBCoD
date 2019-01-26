% Multi-focus Image Fusion Demo script.

% This script demonstrates applying LoBCoD for Multi-Focus image fusion.
% The script loads the following variables:
%  
% (1) Two images to fuse in the variables "Background_inFocus" and
%     "Foreground_inFocus".
% (2) Initial dictionary to represent the edge-components in "D_init".
% (3) Gradient matrices in "Gx" and "Gy" for calculating the gradient in
%     the horizontal and vertical directions, and the natrix 
%     "G = eye + mu*(Gx'*Gx+Gy'*Gy)".
% 

addpath mexfiles;
addpath image_helpers;
addpath('vlfeat/toolbox');
addpath('utilities');
addpath(genpath('spams-matlab'));
vl_setup();

load('datasets/Multi_Focus_example/Multi_Focus_param.mat');
lambda =1;
mu = 5;
n =  sqrt(size(D_init,1));
m = size(D_init,2);
MAXITER_pursuit = 250;

I = cell(1,2);
sz = cell(1,2);

Background_inFocus_lab = rgb2lab(Background_inFocus);
Foreground_inFocus_lab = rgb2lab(Foreground_inFocus);
I_original = rgb2lab(z_bird_rgb);

I{1} = Background_inFocus_lab(:,:,1);
I{2} = Foreground_inFocus_lab(:,:,1);
I_original = double(I_original(:,:,1));

sz{1} = size(I{1});
sz{2} = size(I{2});
sz_vec = sz{1}(1)*sz{1}(2);
N=length(I);
patches = myim2col_set_nonoverlap(I{1}, n);


MAXITER = 2;
Xb = cell(1,N);
X_resb = cell(1,N);
X_res_e = cell(1,N);
alpha =  cell(1,N);
Xe = cell(1,N);
epsilon = 1e-20; 

params = [];
params.lambda = lambda;
params.MAXITER = MAXITER_pursuit;
params.D = D_init;
params.Train_on = false(1);


for k=1:N
    Xe{k} = zeros(size(I{k}));
end
for outerIter = 1 : MAXITER
    for i=1:N
        X_resb{i} = I{i}-Xe{i};
        X_resb{i} = padarray(X_resb{i},[1 1],'symmetric','both');
        Xb{i} = reshape(lsqminnorm(G,X_resb{i}(:)),(sz{i}(1)+2),(sz{i}(2)+2));
        Xb{i} = real(Xb{i}(1:sz{1}(1),1:sz{1}(2)));
        X_res_e{i} = I{i}-Xb{i};
  
    end

    params.Ytrain = X_res_e;
    [Xe,objective,avgpsnr,sparsity,totTime,alpha,~] = LoBCoD(params);
    D_opt = D_init;

end

%% Fusion

A = cell(1,N);
k = (1/14)*ones(14,14);
[feature_maps,~] = create_feature_maps(alpha,n,m,sz{1},D_opt);

fused_feature_maps = cell(1);
fused_feature_maps{1} = cell(size(feature_maps{1}));
Clean_xe = cell(1,length(patches));

A{1} = abs(feature_maps{1}{1});
A{2} = abs(feature_maps{2}{1});
for j=2:m
   A{1} = A{1}+abs(feature_maps{1}{j});
   A{2} = A{2}+abs(feature_maps{2}{j});
end
A{1} = rconv2(A{1},k);
A{2} = rconv2(A{2},k);

for j=1:m
    fused_feature_maps{1}{j} = (A{1}>=A{2}).*feature_maps{1}{j}+(A{1}<A{2}).*feature_maps{2}{j};
end

[alpha_fused,I_rec] = extract_feature_maps(fused_feature_maps,n,m,sz{1},D_opt);
for j=1:n^2 
   Clean_xe{j}= D_opt*alpha_fused{1}{j};
end
       
fused_image_e = mycol2im_set_nonoverlap(Clean_xe,sz{1}, n);
fused_image_b = (A{1}>=A{2}).*Xb{1}+(A{1}<A{2}).*Xb{2};
ours_lab = Foreground_inFocus_lab;
ours_lab(:,:,1)= fused_image_e+fused_image_b;
ours_lab(:,:,2) = (A{1}>=A{2}).*double(Background_inFocus_lab(:,:,2))+(A{1}<A{2}).*double(Foreground_inFocus_lab(:,:,2));
ours_lab(:,:,3) = (A{1}>=A{2}).*double(Background_inFocus_lab(:,:,3))+(A{1}<A{2}).*double(Foreground_inFocus_lab(:,:,3));


% PSNR calculation without the boundaries
PSNR = 20*log10((255*sqrt(numel(I{1}(8:sz{1}-8,8:sz{1}-8))) / norm(reshape(fused_image_e(8:sz{1}-8,8:sz{1}-8)+fused_image_b(8:sz{1}-8,8:sz{1}-8) - I_original(8:sz{1}-8,8:sz{1}-8),1,[]))));
fprintf('PSNR: %.3f\n',PSNR);

figure; 
subplot(2,2,1); imagesc(Background_inFocus); title('Background in-focus'); axis off
subplot(2,2,2); imagesc(Foreground_inFocus); title('Foreground in-focus'); axis off
subplot(2,2,3); imagesc(z_bird_rgb); title('Ground truth'); axis off
subplot(2,2,4); imagesc((lab2rgb(ours_lab))); title(['Our result PSNR: ',num2str(PSNR,4),'dB']); axis off
