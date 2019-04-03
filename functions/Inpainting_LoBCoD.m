function [cleanI,objective,avgpsnr,sparsity,totTime,alpha,D] = Inpainting_LoBCoD(params)
% inpainting - for set of N image
% 
% Inpainting_LoBCoD - Inpainting using the LoBCoD algorithm.
%
% 
% 
%  Usage:
% 
%  [cleanI,objective,avgpsnr,sparsity,totTime,alpha,D] = Inpainting_LoBCoD(params)
% 
% 
%   Main parameters:
%
%       - Y_Original:       The original uncorrupted images.
%       - Y_noisy:          The corrupted images.
%       - lmnI:             The mean of the uncorrupted images.
%       - noisyImean:       The mean of the corrupted images.
%       - M:                The corrupted masks.
%       - lambda:           Regularization parameter.
%       - D:                Initial Local dictionaty (of size n^2 x m).
%                           Dictionary filters are ordered as the columns 
%                           of this matrix.
%
%
%   Optional parameters:
%
%       - MAXITER:          Maximum number of iterations (default 500).
%       - Train_on:         Boolean paramreter, True for training the 
%                           dictionary using Y_noisy, False for performing 
%                           only sparse coding without training D (default False).
% 
%
%   Output:
%       - cleanI:           The reconstructed training images.
%       - objective:        The average objective value on the training set.
%       - avgpsnr:          Average PSNR value calculated over the training set.
%       - sparsity:         The ratio between the number of non-zeros to
%                           the total length of the sparse vector.
%       - totTime:          A time vector containing the iteration timestamps.
%       - alpha:         	The output sparse needles.
%       - D:                The output local dictionaty.
% 
% 
% References:
% "A Local Block Coordinate Descent Algorithm for the Convolutional Sparse Coding Model" E. Zisselman,
% J. Sulam and M. Elad.
% arXiv:1811.00312

%% ----- Constants -------%% 

noiseSD = 0; % Optional

%% ----- Input Parameters -------

if isfield(params,'Y_Original')
    I = params.Y_Original;
else
    error('Original input image missing!\n');
end

if isfield(params,'Y_noisy')
    noisyI = params.Y_noisy;
else
    error('Corrupted input image missing!\n');
end

if isfield(params,'lambda')
    lambda = params.lambda;
else
    error('lambda missing!\n');
end
if isfield(params,'lmnI')
    lmnI = params.lmnI;
else
    error('mean of the uncorrupted images missing!\n');
end

if isfield(params,'noisyImean')
    noisyImean = params.noisyImean;
else
    error('mean of the corrupted image missing!\n');
end

if isfield(params,'M')
    M = params.M;
else
    error('mask missing!\n');
end

if isfield(params,'D')
    D = params.D;
    n =  sqrt(size(D,1));
    m = size(D,2);
else
    error('Initial dictionary missing!\n');
end

if isfield(params,'MAXITER')
    MAXITER = params.MAXITER;
else
    MAXITER = 500;
end

if isfield(params,'Train_on')
    Train_on = params.Train_on;
else
    Train_on = false(1);
end

%% ----- Organize Data -------

N = length(I);
alpha = cell(1,N);
alphaN = cell(1,N);
Clean_x = cell(1,N);
cleanI = cell(1,N); 
resI_patches = cell(1,N);
M_patches = cell(1,N);
sz =  cell(1,N);




%% ----- Sparse Coding Parameters -------


param = [];
%param.L = 6;
param.lambda = lambda;
param.mode = 2;
sumsec     = 0;
tStart = tic;

%% ----- Initialization -------

for i=1:N
    sz{i} = size(I{i}); 
    
    resI_patches{i} = myim2col_set_nonoverlap(noisyI{i}, n);
    M_patches{i} = myim2col_set_nonoverlap(M{i}, n);
    alpha{i} = cell(1,n^2);
    Clean_x{i} = cell(1,n^2);
     
    alphaN{i} = cell2mat(alpha{i});
    
    for j=1:n^2
        resI_patches{i}{j}  = resI_patches{i}{j}/(n^2);
         M_patchesij = M_patches{i}{j};
         resI_patchesij = resI_patches{i}{j};
         alphaij = alpha{i}{j};
        for k=1:size(resI_patches{i}{j},2)
            [ind,~,~] = find(M_patchesij(:,k));
            if (isempty(ind))
                printf('Empty indeces');
                alphaij(:,k) = zeros(m,1);
            else
                val = resI_patchesij(ind,k);
                MD = D(ind,:);
                G = MD'*MD;
                Dt_res = MD'*val;
                alphaij(:,k) = mexLasso(val, G, Dt_res, param);
            end
        end
        alpha{i}{j} = alphaij;
        Clean_x{i}{j}= D*alpha{i}{j};
    end 
    cleanI{i} = mycol2im_set_nonoverlap(Clean_x{i}, sz{i}, n);

    res_I = (noisyI{i}- M{i}.*cleanI{i});
    resI_patches{i} = myim2col_set_nonoverlap(res_I, n);

end

avgpsnr     = zeros(1,MAXITER);
avgpsnr_corrupt = zeros(1,MAXITER);
objective   = zeros(1,MAXITER);
sparsity    = zeros(1,MAXITER);
totTime     = zeros(1,MAXITER);
L1          = zeros(1,MAXITER); 

%% ----- Optimization parameters ------- 

u = 0*D;
mu2 = 0.8;
eta2 = 1e-4;
eta =  0.02;

b1 = 0.99;
b2 = 0.999;
mu = 0*D;
v = 0;
e = 1e-8;

                
tElapsed = toc(tStart);
sumsec = sumsec + tElapsed;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Main Loop                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for outerIter = 1 : MAXITER
    
    sumpsnr     = 0;
    sumnnz      = 0;
    sumnumel    = 0;
    sumpsnr_corrupt = 0;
    suml1          = 0;
    obj         = 0;
    
    tStart = tic;
    
    for i=1:N
        p = randperm(n^2);
        for j=1:n^2
            kk = p(j);
            dxx = floor((kk-0.5)/n)+1; 
            dyy = kk-(floor((kk-0.5)/n))*n;
            cur_sz = sz{i} + 2*(n-1) - [dyy-1, dxx-1];
            padded = padarray((noisyI{i}-M{i}.*cleanI{i}), [n-1,n-1], 'both');
            tmp = padded(dyy:end,dxx:end);
            resI_patches{i}{kk} = im2colstep(tmp, [n,n], [n,n]);
            resI_patchesikk = resI_patches{i}{kk};
            alphaikk = alpha{i}{kk};
            parfor  k=1:size(resI_patches{i}{kk},2)
                
                [ind,~,val] = find(resI_patchesikk(:,k));
                if (isempty(ind))
                    alphaikk(:,k) = zeros(m,1);
                else

                    MD = D(ind,:);
                    res = val + MD*alphaikk(:,k);
                    Dt_res = MD'*res;
                    G = MD'*MD;
                    alphaikk(:,k) = mexLasso(res, G, Dt_res, param);
                end

            end
            
            alpha{i}{kk} = alphaikk;
            res_x = D*alpha{i}{kk}-Clean_x{i}{kk};
            Clean_x{i}{kk} = D*alpha{i}{kk};
            
            dres_I =  zeros(sz{i} + 2*(n-1));
            res_Itemp = col2imstep(res_x, cur_sz, [n,n], [n,n]);
            dres_I(dyy:end, dxx:end) = res_Itemp;
            dres_I = dres_I(n:end-n+1, n:end-n+1);

            cleanI{i} = cleanI{i}+dres_I;
            
            if Train_on    
               resI_patchesik = resI_patches{i}{kk};
               alphaik = alpha{i}{kk};
               grad = - resI_patchesik*alphaik';
                    if outerIter<30
                          mu = b1*mu + (1-b1)*grad;
                          v = b2*v+(1-b2)*(norm(grad(:))^2);
                          mu_hat = mu/(1-b1^((outerIter-1)*n^2+j));
                          v_hat = v/(1-b2^((outerIter-1)*n^2+j));
                          D = D - (eta/(sqrt(v_hat)+e))*mu_hat;
                        else
                          u = mu2*u + eta2*grad;
                          D = D - u;
                     end
                      Dn = diag(1./sqrt(diag(D'*D)));
                      D = D*Dn;
            end

       
        end

        alphaN{i} = cell2mat(alpha{i});
         
         
        sumpsnr = sumpsnr + 20*log10((255*sqrt(numel(I{i}))) / norm(reshape((cleanI{i}+noisyImean{i}) - (I{i}+lmnI{i}),1,[])));
        sumpsnr_corrupt =  sumpsnr_corrupt + 20*log10((255*sqrt(numel(noisyI{i}))) / norm(reshape(M{i}.*cleanI{i} - noisyI{i},1,[])));
        suml1 = suml1+ sum(sum(abs(alphaN{i})));
        sumnnz = sumnnz + nnz(alphaN{i});
        sumnumel = sumnumel + numel(alphaN{i});
        obj = obj + 0.5*norm(noisyI{i} - M{i}.*cleanI{i},'fro')^2 + lambda*sum(sum(abs(alphaN{i})));
        


        
    end   
    

         
    avgpsnr(outerIter) = sumpsnr/N;
    avgpsnr_corrupt(outerIter) = sumpsnr_corrupt/N;
    sparsity(outerIter) = sumnnz/sumnumel;
    objective(outerIter) = obj;
    L1(outerIter)        = suml1;
  
    tElapsed = toc(tStart);
    
    sumsec = sumsec + tElapsed;
    
    totTime(outerIter) = sumsec;
    
    % Print results 
    fprintf('OuterIter = %d, Obj = %.3d, Sparsity = %.3f, Avg-PSNR = %.3f, Avg-PSNR_corr = %.3f, L1 = %.3f, Total-Time (min) = %.3f \n',...
        outerIter,objective(outerIter),sparsity(outerIter),avgpsnr(outerIter),avgpsnr_corrupt(outerIter),L1(outerIter),totTime(outerIter)/60);
    
end

    
end