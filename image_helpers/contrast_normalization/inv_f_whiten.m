%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The function to whiten an image with 1/f whitening.
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2010
%
% @image_file @copybrief inv_f_whiten.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief inf_f_whiten.m
%
% @param img the image to whiten.
%
% @retval whiten_img the whitened image.
% @retval filt the corresponding dwhitening filter that was applied convolutionally.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [whiten_img,filt] = inv_f_whiten(img)

    img = double(img);

  WHITEN_POWER = 4
  WHITEN_SCALE = 0.4
  EPSILON = 1e-6
  
  [imx1,imy1,imc1,M1] = size(img);
  %N=image_size;
  %M=num_images;
  
%   
%   
%   
% ly = imy1;
% lx = imx1;
% sy = imy1;
% sx = imx1;
% 
% %% These values are the index of the small mtx that falls on the
% %% border pixel of the large matrix when computing the first
% %% convolution response sample:
% ctr=1;
% sy2 = floor((sy+ctr+1)/2);
% sx2 = floor((sx+ctr+1)/2);
% 
% % pad:
% img = [ ...
%     img(ly-sy+sy2+1:ly,lx-sx+sx2+1:lx,:,:),...
%     img(ly-sy+sy2+1:ly,:,:,:), ...
% 	img(ly-sy+sy2+1:ly,1:sx2-1,:,:); ...
%     img(:,lx-sx+sx2+1:lx,:,:), ...
%     img, ...
%     img(:,1:sx2-1,:,:); ...
%     img(1:sy2-1,lx-sx+sx2+1:lx,:,:), ...
% 	img(1:sy2-1,:,:,:), ...
% 	img(1:sy2-1,1:sx2-1,:,:) ];

% img = padarray(img,[100 100]);
  
%   keyboard
  
    [imx,imy,imc,M] = size(img);
  whiten_img = zeros(size(img));

  

    [fx fy]=meshgrid(-imy/2:imy/2-1,-imx/2:imx/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=WHITEN_SCALE*mean([imx,imy]);
    filt=rho.*exp(-(rho/f_0).^WHITEN_POWER) + EPSILON;

  for i=1:M
    for c=1:imc
      If=fft2(img(:,:,c,i));
      imagew=real(ifft2(If.*fftshift(filt)));
      whiten_img(:,:,c,i) = imagew;
    end
    
%     If=fftn(img(:,:,:,i));
%     imagew = real(ifftn(If.*fftshift(filt)));
%     whiten_img(:,:,:,i) = imagew;
%     
  end

%   starty = ly-(ly-sy+sy2);
%   startx = lx-(lx-sx+sx2);
%   whiten_img = whiten_img(starty:starty+imy1,startx:startx+imx1,:,:);
%   
  figure(1), sdispims(whiten_img);
  figure(2), sdispims(filt);
  
  %whiten_img(1:2,:,:,:) = EPSILON;
  %whiten_img(:,1:2,:,:) = EPSILON;
  %whiten_img(end-1:end,:,:,:) = EPSILON;
  %whiten_img(:,end-1:end,:,:) = EPSILON;
  
  
  %IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));

  %save MY_IMAGES IMAGES
