%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% The function to dewhiten an image with 1/f whitening.
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2010
%
% @image_file @copybrief inv_f_dewhiten.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% @copybrief inf_f_dewhiten.m
%
% @param img the image to dewhiten.
% @param filt the filter to 
%
% @retval dewhiten_img the whitened image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dewhiten_img] = inv_f_dewhiten(img,filt)

  
  [imx,imy,imc,M] = size(img);
  %N=image_size;
  %M=num_images;

  if nargin==1
    WHITEN_POWER = 4;
    WHITEN_SCALE = 0.4;
    EPSILON = 1e-3;
    [fx fy]=meshgrid(-imy/2:imy/2-1,-imx/2:imx/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=WHITEN_SCALE*mean([imx,imy]);
    filt=rho.*exp(-(rho/f_0).^WHITEN_POWER) + EPSILON;
  end
  
%   img = img(2:end-1,2:end-1,:,:);
%   filt = filt(2:end-1,2:end-1,:,:);
  
    dewhiten_img = zeros(size(img));


  for i=1:M
    for c=1:imc
      If=fft2(img(:,:,c,i));
      imagew=real(ifft2(If./fftshift(filt)));
      dewhiten_img(:,:,c,i) = imagew;
    end

%    If=fftn(img(:,:,:,i));
%     imagew = real(ifftn(If./fftshift(filt)));
%     dewhiten_img(:,:,:,i) = imagew;
%     
  end
  
%   mm = min(dewhiten_img(:));
%   dewhiten_img = dewhiten_img - mm;
%   mx = max(dewhiten_img(:));
%   dewhiten_img = dewhiten_img / mx;
  

  figure(3), sdispims(dewhiten_img,'ycbcr');
  figure(4), sdispims(filt);
