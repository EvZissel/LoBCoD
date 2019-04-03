function [cleanI,objective,avgpsnr,sparsity,totTime,alpha,D] = LoBCoD(params)
% LoBCoD - Local Pursuit and Dictionary update for the CSC model
%
%  This function calculates the sparse vector and updates the dictionary in
%  an batch manner. The function calculates the sparse coding of the training 
%  set and updates the local dictionary.
% 
% 
%   Usage:
% 
%  [cleanI,objective,avgpsnr,sparsity,totTime,alpha,D] = LoBCoD(params)
% 
% 
%   Main parameters:
%
%       - Ytrain:           Training data set, orderd as a three
%                           dimensional array.
%       - lambda:           Regularization parameter.
%       - D:                Initial Local dictionaty (of size n^2 x m).
%                           Dictionary filters are ordered as column 
%                           of this matrix. 
%
%
%   Optional parameters:
%
%       - MAXITER:          Maximum number of iterations (default 500).
%       - Train_on:         Boolean parameter, True for training the 
%                           dictionary using Ytrain, False for performing 
%                           only sparse coding without training D (default True).
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
%       - D:                The output local dictionary. 
% 
% 
% References:
% "A Local Block Coordinate Descent Algorithm for the Convolutional Sparse Coding Model" E. Zisselman,
% J. Sulam and M. Elad.
% arXiv:1811.00312
 

%% ----- Constants -------%% 

noiseSD = 0; %Optional


%% ----- Input Parameters -------

if isfield(params,'Ytrain')
    I = params.Ytrain;
else
    error('Input image missing!\n');
end

if isfield(params,'lambda')
    lambda = params.lambda;
else
    error('\lambda missing!\n');
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
    Train_on = true(1);
end


%% ----- Organize Data -------

N = length(I);
alpha = cell(1,N);
alphaN = cell(1,N);
resI_patchesN = cell(1,N);
Clean_x = cell(1,N);
cleanI = cell(1,N);
resI_patches = cell(1,N);
sz =  cell(1,N);
Valpha = cell(1,N);


%% ----- Sparse Coding Parameters -------


G = D'*D;

param = [];
param.L = 6;
param.lambda = lambda;
param.mode = 2;  
sumsec     = 0;
tStart = tic;

%% ----- Initialization -------


for i=1:N
    sz{i} = size(I{i}); 
     
    noisyI{i} = I{i} + noiseSD * randn(sz{i});
    
    resI_patches{i} = myim2col_set_nonoverlap(noisyI{i}, n);
    alpha{i} = cell(1,n^2);
    Clean_x{i} = cell(1,n^2);
    patchesi =  cell2mat(resI_patches{i});
    patchesi = patchesi/(n^2);
    Valpha{i} = zeros(1,n^2);
     for j=1:n^2 
         Valpha{i}(j) = size(resI_patches{i}{j},2);
     end

    Dt_res = D'*patchesi;
    alphai = mexLasso(patchesi, G, Dt_res, param);
    alpha{i} = mat2cell(alphai,m,Valpha{i});
    Clean_x{i}= mat2cell(D*alphai,n^2,Valpha{i});
    cleanI{i} = mycol2im_set_nonoverlap(Clean_x{i}, sz{i}, n);
    
    
    alphaN{i} = cell2mat(alpha{i});
    res_I = (noisyI{i}-cleanI{i});
    resI_patches{i} = myim2col_set_nonoverlap(res_I, n);
    resI_patchesN{i} = cell2mat(resI_patches{i});
end

avgpsnr     = zeros(1,MAXITER);
objective   = zeros(1,MAXITER);
sparsity    = zeros(1,MAXITER);
totTime     = zeros(1,MAXITER);


%% ----- Optimization parameters ------- 

u = 0*D;
mu2 = 0.8;
eta =  0.022;
eta2 = 3e-7;

num_steps = 1;3;5;
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
    obj         = 0;
    
    tStart = tic;
    
    pi = randperm(N);
    
    for ii=1:N
        i = pi(ii);
        p = randperm(n^2);
        for j=1:n^2
           k = p(j); %Layers are chosen at random order
           dx = floor((k-0.5)/n)+1; 
           dy = k-(floor((k-0.5)/n))*n;
           cur_sz = sz{i} + 2*(n-1) - [dy-1, dx-1];
           padded = padarray((noisyI{i}-cleanI{i}), [n-1,n-1], 'both');
           tmp = padded(dy:end,dx:end);
           resI_patches{i}{k} = im2colstep(tmp, [n,n], [n,n]);
           res = resI_patches{i}{k}+D*alpha{i}{k};
           Dt_res = D'*res;
           alpha{i}{k} = mexLasso(res, G, Dt_res, param);

           res_x = D*alpha{i}{k}-Clean_x{i}{k};
           Clean_x{i}{k} = D*alpha{i}{k};

           dres_I =  zeros(sz{i} + 2*(n-1));
           res_Itemp = col2imstep(res_x, cur_sz, [n,n], [n,n]);
           dres_I(dy:end, dx:end) = res_Itemp;
           dres_I = dres_I(n:end-n+1, n:end-n+1);

           cleanI{i} = cleanI{i}+dres_I;

             
           padded = padarray((noisyI{i}-cleanI{i}), [n-1,n-1], 'both');
           tmp = padded(dy:end,dx:end);
           resI_patches{i}{k} = im2colstep(tmp, [n,n], [n,n]);

           if (Train_on) 
               for t=1:num_steps 

                  resI_patchesik = resI_patches{i}{k};
                  alphaik = alpha{i}{k};
                  grad = - resI_patchesik*alphaik';

                  if (outerIter<40)
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

                G = D'*D;
          end
         
        end

        alphaN{i} = cell2mat(alpha{i});

        sumpsnr = sumpsnr + 20*log10((255*sqrt(numel(I{i}))) / norm(reshape(cleanI{i} - I{i},1,[])));
        sumnnz = sumnnz + nnz(alphaN{i});
        sumnumel = sumnumel + numel(alphaN{i});
        obj = obj + 0.5*norm(noisyI{i} - cleanI{i},'fro')^2 + lambda*sum(sum(abs(alphaN{i})));
   
        
        
    end   
    
    % Replace unused atoms
    if (Train_on)
        alpha_flat = cell2mat(alphaN);
        unused_sigs= 1 : size(alphaN{i},2);
        replaced_atoms = zeros(1,m);
        for i = 1 : m   

            idxs = find(alpha_flat(i,:)); %find all the examples
            if length(idxs) < 1
                maxsignals = 500;
                perm = randperm(length(unused_sigs));
                perm = perm(1:min(maxsignals,length(perm)));

                E = patchesi(:,unused_sigs(perm))- D*alpha_flat(:,unused_sigs(perm));
                E = sum(E.^2);
                [~,maxidx] = max(E);

                D(:,i) = patchesi(:,unused_sigs(perm(maxidx)));
                D(:,i) = D(:,i) / norm(D(:,i));
                if isnan(norm(D(:,i))<1e-10)
                   disp('The norme of di is zero-SBDL_lambda',num2str(lambda),'_PatchSize',num2str(n),'_m',num2str(m))
                   return
                end

                unused_sigs = unused_sigs([1:perm(maxidx)-1,perm(maxidx)+1:end]);

                replaced_atoms(i) = 1;

                fprintf('replaced atom\n');

                G = D'*D;

                continue;
            end

        end
    end
         
    avgpsnr(outerIter) = sumpsnr/N;
    sparsity(outerIter) = sumnnz/sumnumel;
    objective(outerIter) = obj;
  
  
    tElapsed = toc(tStart);
    
    sumsec = sumsec + tElapsed;
    
    totTime(outerIter) = sumsec;
    
    %Print results 
    if (Train_on) 
     fprintf('OuterIter = %d, Obj = %.3d, Sparsity = %.3f, Avg-PSNR = %.3f, Total-Time (min) = %.3f \n',...
        outerIter,objective(outerIter),sparsity(outerIter),avgpsnr(outerIter),totTime(outerIter)/60);
    end
    
end


    
end