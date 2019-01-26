%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This functin is used to whiten an image with patch based ZCA whitening that is applied
% convolutionally to the image.
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2010
%
% @image_file @copybrief region_zca.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief region_zca.m
%
% @param img the image to whiten.
%
% @retval w_filt the whitening filter to apply convolutionally.
% @retval d_filt the corresponding de-whitening filter to apply convolutionally.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w_filt,d_filt] = region_zca(img)

NUM_PATCHES = 1;
PATCH_SIZE = 19;
BORDER_SIZE = 4;
[imx,imy,imc,M] = size(img);

count = 0;
w_filt = zeros(PATCH_SIZE,PATCH_SIZE,3,3);
d_filt = zeros(PATCH_SIZE,PATCH_SIZE,3,3);

for i=1:NUM_PATCHES
    i
    %% choose start coords of patch
    r = floor(rand(1,2).*([imy,imx]-PATCH_SIZE)+1);
    
    %% grab patches & compute covariance matrix
    p = img(r(1):r(1)+PATCH_SIZE-1,r(2):r(2)+PATCH_SIZE-1,:,:);
    m = reshape(p,[3*PATCH_SIZE^2,M]);
    mu = mean(m,2);
    cm = double(m) - mu*ones(1,M);
    CC = cm * cm';
    
    %[u,s,v] = svd(CC);
    [u,s] = eig(CC);
    
    q = diag(s);
    qi = find(q>0);
    q1(qi) = q(qi).^-0.5;
    q2(qi) = q(qi).^0.5;
    
    W = u(:,qi)*diag(q1(qi))*u(:,qi)';
    D = u(:,qi)*diag(q2(qi))*u(:,qi)';
    
    for d=1:imc
        for j=[(d-1)*floor(size(W,1)/3)+1:d*floor(size(W,1)/3)]
            r = reshape(W(j,:),[PATCH_SIZE PATCH_SIZE 3]);
            dr = reshape(D(j,:),[PATCH_SIZE PATCH_SIZE 3]);
            [sy,sx] = ind2sub([1 1]*PATCH_SIZE,j-(d-1)*floor(size(W,1)/3));
            sx = -(sx - floor(PATCH_SIZE/2));
            sy = -(sy - floor(PATCH_SIZE/2));
            
            if (abs(sx)<floor(PATCH_SIZE/2)-BORDER_SIZE) &  (abs(sy)<floor(PATCH_SIZE/2)-BORDER_SIZE)
                for c=1:imc
                    r2(:,:,c) = circshift(r(:,:,c),[sy sx]+1);
                    dr2(:,:,c) = circshift(dr(:,:,c),[sy sx]+1);
                end
                w_filt(:,:,:,d) = w_filt(:,:,:,d) + r2;
                d_filt(:,:,:,d) = d_filt(:,:,:,d) + dr2;
                count = count + 1;
                % figure(99); imagesc(uint8(1e-1*dr2)); pause(0.1); drawnow;
                
            end
            
        end
    end
    
    figure(1); clf; montage(reshape(w_filt/sum(w_filt(:).^2),PATCH_SIZE,PATCH_SIZE,1,imc^2)); caxis([-0.1 0.1]);
    figure(2); clf; montage(reshape(d_filt/sum(d_filt(:).^2),PATCH_SIZE,PATCH_SIZE,1,imc^2)); caxis([-1e-3 1e-3]);
    
    
    %keyboard
    
    %%% check conv vs patch based efforts
    for d=1:3
        im_w(:,:,d)=convn(img(:,:,:,1),w_filt(:,:,3:-1:1,d),'valid');
    end
    for d=1:3
        im_dw(:,:,d)=convn(im_w,d_filt(:,:,3:-1:1,d),'valid');
        filt_check(:,:,:,d)=convn(w_filt(:,:,3:-1:1,d),d_filt(:,:,3:-1:1,d),'same');
    end
    figure(6); clf; montage(reshape(filt_check,PATCH_SIZE,PATCH_SIZE,1,imc^2)); caxis([-1e3 1e3]);
    
    im_w = im_w - min(im_w(:));
    im_w = im_w / max(im_w(:));
    
    im_dw = im_dw - min(im_dw(:));
    im_dw = im_dw / max(im_dw(:));
    
    q = [];
    for c=1:3
        q= [ q ; im2col(img(:,:,c,1),[1 1]*PATCH_SIZE,'distinct') ];
    end
    block_w = W*double(q);
    for c=1:3
        im_w2(:,:,c) = col2im(block_w((c-1)*PATCH_SIZE^2+1:c*PATCH_SIZE^2,:),[1 1]*PATCH_SIZE,[imy imx],'distinct');
    end
    
    
    q = [];
    for c=1:3
        q= [ q ; im2col(im_w2(:,:,c),[1 1]*PATCH_SIZE,'distinct') ];
    end
    block_dw = D*double(q);
    for c=1:3
        im_dw2(:,:,c) = col2im(block_dw((c-1)*PATCH_SIZE^2+1:c*PATCH_SIZE^2,:),[1 1]*PATCH_SIZE,[imy imx],'distinct');
    end
    
    im_w2 = im_w2 - min(im_w2(:));
    im_w2 = im_w2 / max(im_w2(:));
    
    figure(3); clf;
    ultimateSubplot(1,2,1,1,0.1); imagesc(uint8(255*im_dw));
    ultimateSubplot(1,2,1,2,0.1); imagesc(uint8(im_dw2));
    
    figure(4); clf;
    ultimateSubplot(1,2,1,1,0.1); imagesc(uint8(255*im_w));
    ultimateSubplot(1,2,1,2,0.1); imagesc(uint8(255*im_w2));
    
    
end
w_filt = w_filt(:,:,3:-1:1,:) / (count/3);
d_filt = d_filt(:,:,3:-1:1,:) / (count/3);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve least squares system for the dewhitening fitlers.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_filt2 = zeros(size(d_filt));
% Get middle index (where the filters are in ZCAtransform.
middle = sub2ind([PATCH_SIZE PATCH_SIZE],ceil(PATCH_SIZE/2),ceil(PATCH_SIZE/2));
for d=1:size(w_filt,4) %% The output channels.
    for c=1:size(w_filt,3) % The input image color channels.
        W = conv2mat(w_filt(:,:,c,d),[PATCH_SIZE PATCH_SIZE],'full');
        rhs = zeros(size(W,1),1);
        rhs(middle) = 1;
        d_filt2(:,:,c,d) = reshape(W\rhs,PATCH_SIZE,PATCH_SIZE);
    end
    figure(200+d)
    imshow(d_filt2(:,:,:,d));
%     keyboard
end

for c=1:3
    filt_check2(:,:,:,c)=convn(w_filt(:,:,3:-1:1,c),d_filt2(:,:,3:-1:1,c),'same');
end
figure(7); clf; montage(reshape(filt_check2,PATCH_SIZE,PATCH_SIZE,1,imc^2)); caxis([-1e3 1e3]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






keyboard
