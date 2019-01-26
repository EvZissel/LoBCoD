%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This takes all images from the input folder, converts them to the desired
% colorspace, removes mean/divides by standard deviations (if desired), and
% constrast normalizes the image (if desired). If the images are of different
% sizes, then it will padd them with zeros (after contrast normalizing) to make
% them square (assumes that they all images have the same maximum dimension).
% Note that some of the whitening/contrast normalization features are not
% fully tested for datasets where the images are of variable size so please
% use with caution in that case. For best result, resize all the images to the
% same dimensions beforehand.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @image_file @copybrief CreateImages.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief CreateImages.m
%
% @param imgs_path either 1) a path to a folder contain only image files (or other folders
% which will be ignored), 2) a variable with the images as xdim x ydim x
% num_colors x num_images size , 3) a path to a file that contains a variable
% called I that has images as xdim x ydim x num_colors x num_images, or 4)
% a path to a folder containing a single .mat file containing a variable called
% I that has images as xdim x ydim x num_colors x num_images.
% @param CONTRAST_NORMALIZE [optional] binary value indicating whether to contrast
% normalize or whiten the images. Defaults to local contrast normalization ('local_cn').
% Available types are: 'none','local_cn','laplacian_cn','box_cn','PCA_whitening',
% 'ZCA_image_whitening','ZCA_patch_whitening',and 'inv_f_whitening'
% @param ZERO_MEAN [optional] binary value indicating whether to subtract the mean and divides by standard deviation (current
% commented out in the code). Defuaults to 1.
% @param COLOR_TYPE [optional] a string of: 'gray','rgb','ycbcr','hsv'. Defaults to 'gray'.
% @param SQUARE_IMAGES [optional] binary value indicating whether or not to square the
% images. This must be used if using different sized images. Even then the max
% dimensions of each image must be the same. Defaults to 0.
% @param image_selection the subset of images you want to select. This is a cell
% array with 3 dimensions, {A,B,C} -> A:B:C where A and B are numbers and C can
% be a number or 'end' string.
%
% @retval I the images as: xdim x ydim x color_channels x num_images
% @retval mn the mean if ZERO_MEAN was set.
% @retval sd the standard deviation if ZERO_MEAN was set.
% @retval xdim the size of the images in x direction.
% @retval ydim the size of the images in y direction.
% @retval resI the (image-contrast normalized image) if CONTRAST_NORMALIZE is
% set.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I] = CreateImages(imgs_path,CONTRAST_NORMALIZE,ZERO_MEAN,COLOR_TYPE,SQUARE_IMAGES,image_frames)

% Defaults
if(nargin<6)
    image_frames = {1,1,'end'};
end

if(nargin<5)
    SQUARE_IMAGES = 0;
end

if(nargin<4)
    COLOR_TYPE = 'gray'
end

if(nargin<3)
    ZERO_MEAN = 1
end

if(nargin<2)
    CONTRAST_NORMALIZE = 'local_cn'
end

if(isnumeric(CONTRAST_NORMALIZE))
    if(CONTRAST_NORMALIZE==1)
        CONTRAST_NORMALIZE = 'local_cn';
    else
        CONTRAST_NORMALIZE = 'none';
    end
end


% For backwards compatibility, revert to grayscale.
if(isnumeric(COLOR_TYPE))
    if(COLOR_TYPE == 1)
        COLOR_TYPE = 'rgb';
    else
        COLOR_TYPE = 'gray';
    end
end

% For backwards compatibility, revert to grayscale.
if(isnumeric(COLOR_TYPE))
    if(COLOR_TYPE == 1)
        COLOR_TYPE = 'local_cn';
    else
        COLOR_TYPE = 'none';
    end
end

% Select only the frames (images) you want.
if(ischar(image_frames{3}))
    last_ind = sprintf('%d:%d:%s',image_frames{1},image_frames{2},image_frames{3});
else
    last_ind = sprintf('%d:%d:%d',image_frames{1},image_frames{2},image_frames{3});
end

fprintf('Going to select frames: %s',last_ind);

% Cell array listing all files and paths.
subdir = dir(imgs_path);
[~,files] = split_folders_files(subdir);

if(check_imgs_path(imgs_path)==0)
    error('Path to images is not a valid .mat file or a directory of images.');
end


% Make sure it is a directory and doesn't just have one file that is not an image.
if(ischar(imgs_path) && exist(imgs_path,'dir')>0 && (length(files)>1 ...
        || (length(files)==1 && strcmp(files(1).name(end-3:end),'.mat')==0)))
    
    % Make sure the directory ends in '/'
    if(strcmp(imgs_path(end),'/')==0)
        imgs_path = [imgs_path '/'];
    end
    
    % Counter for the image
    image = 1;
    
    if(length(files) == 0)
        error('No Images in this directory');
    end
    
    
   % I = cell(1,length(files));
    actual_files = 0;
    
    fprintf('The length of the I file cell array found in this directory is: %d\n',length(files));
    
    
    % Make sure the selection is not over the number of files.
    if(ischar(image_frames{3}))
        image_frames{3} = length(files);
    else
        if(image_frames{3}>length(files))
            image_frames{3}=length(files);
        end
    end
    
    
    % Loop through the number of files ignoring . and ..
    for file=image_frames{1}:image_frames{2}:image_frames{3}
        % Makes sure not to count subdirectories
        if (files(file).isdir == 0)
            
            
            % Get the path to the given file.
            img_name = strcat(imgs_path,files(file).name);
            
            try
                % Load the image file
                IMG = single(imread(img_name));
                
                % Count number of images loaded.
                actual_files = actual_files+1;
                fprintf('Loading: %s \n Image: %10d/%10d. Selecting every %5d. Selected: %10d so far.\r',img_name,file,length(files),image_frames{2},actual_files);
                
                if(actual_files==1)
                    I = cell(1,length(files));
                end
                I{actual_files} = IMG;
                % Increment the number of images found so far.
                image=image+1;
            catch
                fprintf('Counld not load %s as an image.\n',img_name);
            end
        end
    end
    I = I(1:actual_files);
    
    % Automatically find the .mat file in the directory that contains all images (no other files or folders can be in teh directory though).
elseif(ischar(imgs_path) && (length(files)==1 && strcmp(files(1).name(end-3:end),'.mat'))) % Only 1 file in folder and it's a .mat
    
    load([imgs_path files(1).name]);
            
    if(exist('original_images','var'))
        I = original_images;
        clear original_images;
    end
    clear imgs_path
    % Make sure the images are single.
    I = single(I);
    
    
    % Select the ones you want.
    eval(strcat('I = I(:,:,:,',last_ind,');'));
    origI = I;
    I = cell(1,size(origI,4));
    for i=1:size(origI,4)
        fprintf('Loaded image: %10d from file.\r',i);
        I{i} = origI(:,:,:,i);
    end
    actual_files = size(origI,4);
    clear origI
elseif(ischar(imgs_path) && exist(imgs_path,'file')~=0)  % Path is to a file with all images in it.
    fprintf('\nLoading %s\n',imgs_path);
    % May be a .mat file.
    if(strcmp(imgs_path(end-3:end),'.mat'))
        fprintf('Loading single .mat file');
        load(imgs_path);
    else % May be a single image.
        fprintf('Reading single image.\n');
        I = single(imread(imgs_path));
    end
    if(exist('original_images','var'))
        I = original_images;
        clear original_images;
    end
    clear imgs_path
    I = single(I);
    % Select the ones you want.
    eval(strcat('I = I(:,:,:,',last_ind,');'));
    fprintf('Converting from matrix to cell array selected %d images.\n',size(I,4));

    I = mat2cell(I,size(I,1),size(I,2),size(I,3),ones(size(I,4),1));
    I = reshape(I,[size(I,4) 1]);
    actual_files = size(I,2);    
else % The imgs_path is a variable of images ( can just use it instead of loading).
    
    I = imgs_path;
    clear imgs_path
    
    % Select the ones you want.
    eval(strcat('I = I(:,:,:,',last_ind,');'));
    
    origI = I;
    I = cell(1,size(origI,4));
    for i=1:size(origI,4)
        fprintf('Loaded image: %10d from file.\r',i);
        I{i} = origI(:,:,:,i);
    end
    actual_files = size(origI,4);
    clear origI
end
clear files subdir
fprintf('\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Convert the colors here.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% old_I = I;
for i=1:length(I)
    switch(COLOR_TYPE)
        case 'rgb'
            fprintf('Making RGB Image %10d\r',i);
            %             IMG = rgb_im;
            % Normalize the RGB values to [0,1] (do not do this on YCbCr!!!!!).
            I{i} = double(I{i})/255.0;
        case 'ycbcr'
            fprintf('Making YUV Image %10d\r',i);
            I{i} = double(rgb2ycbcr(double(I{i})/255.0));
        case 'hsv'
            fprintf('Making HSV Image %10d\r',i);
            I{i} = double(rgb2hsv(double(I{i})/255.0));
        case 'gray'
            fprintf('Making Gray Image %10d\r',i);
            % Convert to grayscale
            if(size(I{i},3)==3)
                I{i} = single(rgb2gray(double(I{i})/255.0));
            else
                if(max(I{i}(:))>1)
                    I{i} = double(I{i})/255.0;
                else
                    I{i} = double(I{i});
                end
            end
    end
    
    %     I{i} = IMG;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Contrast normalize the image?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CN_I = cell(size(I));
% res_I = cell(size(I));
switch(CONTRAST_NORMALIZE)
    
    case 'none'
        fprintf('Not doing any constrast normalization or whitening to the images.\n');
        % Make the CN_I result the original images.
        %         CN_I = I;
        %         res_I = I;
        
    case 'local_cn'
        %%%%%
        %% Local Constrast Normalization.
        %%%%%
        num_colors = size(I{1},3);
        %         k = fspecial('gaussian',[13 13],1.591*3);
        %         k = fspecial('gaussian',[5 5],1.591);
        k = fspecial('gaussian',[13 13],3*1.591);
        k2 = fspecial('gaussian',[13 13],3*1.591);
%         k = fspecial('gaussian',[7 7],1.5*1.591);
%         k2 = fspecial('gaussian',[7 7],1.5*1.591);
        if(all(k(:)==k2(:)))
            SAME_KERNELS=1;
        else
            SAME_KERNELS=0;
        end
        
        for image=1:length(I)
            fprintf('Contrast Normalizing Image with Local CN: %10d\r',image);
            temp = I{image};
            for j=1:num_colors
                %                 if(image==151)
                %                     keyboard
                %                 end
                dim = double(temp(:,:,j));
                %                 lmn = conv2(dim,k,'valid');
                %                 lmnsq = conv2(dim.^2,k,'valid');
                lmn = rconv2(dim,k);
                lmnsq = rconv2(dim.^2,k2);
                if(SAME_KERNELS)
                    lmn2 = lmn;
                else
                    lmn2 = rconv2(dim,k2);
                end
                lvar = lmnsq - lmn2.^2;
                lvar(lvar<0) = 0; % avoid numerical problems
                lstd = sqrt(lvar);
                
                q=sort(lstd(:));
                lq = round(length(q)/2);
                th = q(lq);
                if(th==0)
                    q = nonzeros(q);
                    if(~isempty(q))
                    lq = round(length(q)/2);
                    th = q(lq);
                    else
                        th = 0;
                    end
                end
                lstd(lstd<=th) = th;
                %lstd(lstd<(8/255)) = 8/255;
                %                 lstd = conv2(lstd,k2,'same');
                
                
                lstd(lstd(:)==0) = eps;
                
                %                 shifti = floor(size(k,1)/2)+1;
                %                 shiftj = floor(size(k,2)/2)+1;
                
                % since we do valid convolutions
                %                 dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
                dim = dim - lmn;
                dim = dim ./ lstd;
                
                temp(:,:,j) = dim;
                %                 res_I{image}(:,:,j) = single(double(I{image}(:,:,j))-dim);
                %                 res_I{image}(:,:,j) = double(I{image}(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
                %             IMG = conI;
            end
            I{image} = single(temp);
        end
    case 'laplacian_cn'
        %%%%%
        %% CN with a laplacian filter (CVPR 2010 method).
        %%%%%
        % Run a laplacian over images to get edge features.
        h = fspecial('laplacian',0.2);
        
        shifti = floor(size(h,1)/2)+1;
        shiftj = floor(size(h,2)/2)+1;
        % Loop through the number of images
        for image=1:length(I)
            fprintf('Contrast Normalizing Image with Laplacian: %10d\r',image);
            for j=1:size(I{1},3)  % Each color plane needs to be passed with laplacin
                I{image}(:,:,j) = conv2(single(I{image}(:,:,j)),single(h),'same');
                %                 res_I{image}(:,:,j) = double(I{image}(shifti:shifti+size(CN_I{image},1)-1,shiftj:shiftj+size(CN_I{image},2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
            end
        end
    case 'box_cn'
        %%%%%%
        %% CN with a box filter (has bad boundary effects though)
        %%%%%%
        boxf = ones(5,5)/25;
        for image=1:size(I,4)
            fprintf('Contrast Normalizing Image with Box Filtering: %10d\r',image);
            for j=1:size(I,3)
                I(:,:,j,image) = I(:,:,j,image) - imfilter(I(:,:,j,image),boxf,'replicate');
            end
            CN_I{image} = I(:,:,:,image);
        end
    case 'PCA_whitening'
        %%%%%
        %% PCA based whitening
        %%%%%
        for color=1:size(I,3)
            fprintf('\nPCA whitening all images...\n\n');
            
            data = double(reshape(I(:,:,color,:),size(I,1)*size(I,2),size(I,4)));
            
            % size(data)
            % center the data
            % Only take mean if more than one image.
            if(ZERO_MEAN==0)
                fprintf('Taking zero mean of the dataset anyways.\n')
                if(size(data,2)>1)
                    mn = mean(data,2);
                else
                    mn=mean(data(:));
                end
                data = data - repmat(mn,1,size(data,2));
                sd = std(data(:));
                data = data/sd;
            end
            cc = cov(data);
            [V D] = eig(cc);
            
            ii = cumsum(fliplr(diag(D)'))/sum(D(:));
            nrc = length(find(ii<0.99)); % retain 99% of the variance
            V = V(:,end-nrc+1:end);
            D = D(end-nrc+1:end,end-nrc+1:end);
            PCAtransf = diag(diag(D).^-0.5) * V';
            invPCAtransf = V * diag(diag(D).^0.5);
            data = single(data * PCAtransf');
            % whitendata = single(data * PCAtransf');
            I(:,:,color,1:size(PCAtransf,1)) = reshape(data,size(I(:,:,color,1:size(PCAtransf,1))));
        end
        for image=1:size(I,4)
            CN_I{image} = I(:,:,:,image);
        end
    case 'ZCA_image_whitening'
        %%%%%
        %% ZCA image based whitening (uses entire images).
        %% this is much slower than the below for large images.
        %%%%%
        fprintf('\nZCA whitening all images...this can take a while...\n\n');
        
        data = double(reshape(I,size(I,1)*size(I,2)*size(I,3),size(I,4)));
        
        % size(data)
        % center the data
        % Only take mean if more than one image.
        if(ZERO_MEAN==0)
            fprintf('Taking zero mean of the dataset anyways.\n')
            if(size(data,2)>1)
                mn = mean(data,2);
            else
                mn=mean(data(:));
            end
            data = data - repmat(mn,1,size(data,2));
            sd = std(data(:));
            data = data/sd;
        end
        cc = cov(data');
        [V D] = eig(cc);
        indx = find(diag(D) > 0);
        ZCAtransform = V(:,indx) * inv(sqrt(D(indx,indx))) * V(:,indx)';
        invZCAtransform = V(:,indx) * sqrt(D(indx,indx)) * V(:,indx)';
        % whitening happens here.
        data = data*ZCAtransform;
        
        %     data = data*invZCAtransform*sd+repmat(mn,1,size(data,2));
        I = reshape(data,size(I));
        
        for image=1:size(I,4)
            CN_I{image} = I(:,:,:,image);
        end
    case 'ZCA_patch_whitening'
        %%%%%
        %% ZCA patch based whitening (uses randomly selected patches)
        %% this is much faster than the above.
        %%%%%
        %%%%%%%%%%%%%%%%%%%
        % Define the patch size (largest one possible)
        %%%%%%%%%%%%%%%%%%%
        for patch_size=size(I,1):-1:1
            % Has to evenly divide into image.
            if(mod(size(I,1),patch_size)==0)
                temp = im2col(I(:,:,1,1),[patch_size patch_size],'distinct');
                % Need more patches from the dataset than size of
                % patches (which are times # of colors).
                if(size(temp,2)*size(I,4)>size(temp,1)*size(I,3))
                    break
                end
            end
        end
        fprintf('Size of the whitening filter is %d.\n',patch_size);
        %%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%
        % Derive whitening transform from random patches of the images.
        %%%%%%%%%%%%%%%%%%%
        temp = im2col(I(:,:,1,1),[patch_size patch_size],'distinct');
        data = zeros(patch_size^2,size(temp,2),size(I,3),size(I,4));
        clear temp
        
        % Create image patches of 11x11 size.
        indices = randperm(size(I,4));
        for i=1:size(I,4)
            % Use random selection of the images.
            ind = indices(i);
            for color=1:size(I,3)
                data(:,:,color,i) =  im2col(I(:,:,color,ind),[patch_size patch_size],'distinct');
            end
            % Keep only 100,000 patches around for computing the whitening transforms.
            if(size(data,2)*i>100000)
                break
            end
        end
        fprintf('\nZCA whitening all images based on patches...\n\n');
        
        data = data(:,:,:,1:i);
        data=permute(data,[1 3 2 4]);
        [patch colors num_patches num_images] = size(data)
        
        data = reshape(data,size(data,1)*size(data,2),size(data,3)*size(data,4));
        
        patch_mn = mean(data,2);
        data = data - repmat(patch_mn,[1 size(data,2)]);
        patch_sd = std(data(:));
        data = data/patch_sd;
        
        size(data)
        
        cc = cov(data');
        [V D] = eig(cc);
        indx = find(diag(D) > 0);
        ZCAtransform = V(:,indx) * inv(sqrt(D(indx,indx))) * V(:,indx)';
        invZCAtransform = V(:,indx) * sqrt(D(indx,indx)) * V(:,indx)';
        
        % Get middle index (where the filters are in ZCAtransform.
        middle = sub2ind([patch_size patch_size],ceil(patch_size/2),ceil(patch_size/2));
        %%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%
        % Show whitening filters
        %%%%%%%%%%%%%%%%%%%
        %for color=1:size(I,3)
        %    filters(:,:,:,color) = reshape(ZCAtransform((color-1)*patch_size^2+middle,:)',patch_size,patch_size,colors);
        %    figure(100+color)
        %    imshow(filters(:,:,:,color))
        %end
        %%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%
        % Show whitening filters
        %%%%%%%%%%%%%%%%%%%
        %for color=1:size(I,3)
        %    filters2(:,:,:,color) = reshape(invZCAtransform((color-1)*patch_size^2+middle,:)',patch_size,patch_size,colors);
        %    figure(200+color)
        %    imshow(filters2(:,:,:,color))
        %end
        %%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%
        % Applying ZCA transform to each distinct patch and then forming images
        % again.
        clear data
        for i=1:size(I,4)
            for color=1:size(I,3)
                data(:,:,color,i) =  im2col(I(:,:,color,i),[patch_size patch_size],'distinct');
            end
        end
        data=permute(data,[1 3 2 4]);
        [patch colors num_patches num_images] = size(data);
        data = reshape(data,size(data,1)*size(data,2),size(data,3)*size(data,4));
        
        data= ZCAtransform*data;
        data = reshape(data,patch,colors,num_patches,num_images);
        data=permute(data,[1 3 2 4]);
        
        for i=1:size(I,4)
            for color=1:size(I,3)
                I(:,:,color,i) =  col2im(data(:,:,color,i),[patch_size patch_size],[size(I,1) size(I,2)],'distinct');
            end
        end
        %%%%%%%%%%%%%%%%%%%%
        for image=1:size(I,4)
            CN_I{image} = I(:,:,:,image);
        end
    case 'inv_f_whitening'
        %%%%%
        %% 1/f whitening of the images
        %%%%%
        % Number of images.
        M=length(I);
        
        REGULARIZATION=0.3;
        WHITEN_POWER = 4;
        WHITEN_SCALE = 0.4;
        EPSILON = 1e-3;
        BORDER=0;
        
        for i=1:M
            
            fprintf('Whitening image: %10d\r',i);
            temp_im = I{i};
            
            [imx,imy,imc] = size(temp_im);
            if(exist('I','var')==0)
                I = zeros(imx,imy,imc,M);
            end
            
            % Make 1/f filter
            [fx fy]=meshgrid(-imy/2:imy/2-1,-imx/2:imx/2-1);
            rho=sqrt(fx.*fx+fy.*fy)+REGULARIZATION;
            f_0=WHITEN_SCALE*mean([imx,imy]);
            filt=rho.*exp(-(rho/f_0).^WHITEN_POWER) + EPSILON;
            
            for c=1:imc
                If=fft2(temp_im(:,:,c));
                imagew=real(ifft2(If.*fftshift(filt)));
                
                BORDER_VALUE = mean(imagew(:));
                
                if(BORDER~=0)
                    imagew(1:BORDER,:,:,:)=BORDER_VALUE;
                    imagew(:,1:BORDER,:,:)=BORDER_VALUE;
                    imagew(end-BORDER+1:end,:,:,:)=BORDER_VALUE;
                    imagew(:,end-BORDER+1:end,:,:)=BORDER_VALUE;
                end
                
                temp_im(:,:,c) = imagew;
                
            end
            CN_I{i} = temp_im;
            res_I{i} = I{i}-CN_I{i};
            
            %             I(:,:,:,i) = temp_im;
        end
    case 'sep_mean' % Make each image separately have zero mean (useful for text.
        for i=1:length(I)
            fprintf('Zero Meaning Image %10d',i);
            I{i} = I{i}-mean(I{i}(:));
            %             res_I{i} = 0;
        end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');



if ZERO_MEAN
    for i=1:length(I)
        fprintf('Making Image %10d Zero Mean.\r',i);
        I{i} = I{i} - mean(I{i}(:));
    end
end
fprintf('\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Square the images to the max dimension.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear PADIMG;  % Size of I may have changed by this point due to CN.
if(SQUARE_IMAGES)
    % Now pad them again to ensure they are square.
    % This has to be done after contrast normalizing to avoid strong edges on padded
    % regions.
    %     max_size = max(size(I{1},1),size(I{1},2));
    %     I = zeros(max_size,max_size,size(I{1},3),length(I),'single');
    %     resI = zeros(max_size,max_size,size(I{1},3),length(I),'single');
    for image=1:length(I)
        [xdim ydim planes] = size(I{image});
        if(xdim~=ydim) % If not already square.
            maxdim = max(xdim,ydim);
            PADIMG = zeros(maxdim,maxdim,planes,'single');
            %             RESIMG = zeros(maxdim,maxdim,planes,'single');
            for plane=1:planes
                tempimg = padarray(I{image}(:,:,plane),[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2)],'pre');
                PADIMG(:,:,plane) = padarray(tempimg,[ceil((maxdim-xdim)/2) ceil((maxdim-ydim)/2)],'post');
                %                 tempimg = padarray(res_I{image}(:,:,plane),[floor((maxdim-xdim)/2) floor((maxdim-ydim)/2)],'pre');
                %                 RESIMG(:,:,plane) = padarray(tempimg,[ceil((maxdim-xdim)/2) ceil((maxdim-ydim)/2)],'post');
            end
            % Store the padded images into a matrix (as they are all the same
            % dimension).
            fprintf('Squaring Image: %10d\r',image);
            I{image} = single(PADIMG);
            %             resI(:,:,:,image) = RESIMG;
        else
            fprintf('Image %10d Already Square\r',image);
            I{image} = single(I{image});
            %             resI(:,:,:,image) = res_I{image};
        end
        % Save memory.
        %         I{image} = [];
        %         res_I{image} = [];
    end
    
end


% Now all of I is assumed to be the same size.
[xdim ydim colors] = size(I{1});
numims = length(I);
% Make sure it is a row vector.
I = reshape(I,[1 numims]);

I = single(cell2mat(I));
I = reshape(I,[xdim ydim numims colors]);
I = permute(I,[1 2 4 3]);
I = double(I);

%     I = reshape(I,[xdim ydim colors numims]);


fprintf('Not Squaring, just converting all images from cell to matrix...\n')
%     for image=1:length(I)
%         I(:,:,:,image) = single(I{image});
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nAll Images have been loaded and preprocessed.\n\n');


