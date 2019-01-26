%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This function is to unwhiten a whitened image in the fourier domain.
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2010
%
% @image_file @copybrief unwhiten.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
%   N=image_size;
% N=100;
%   M=num_images;
%   M=1;

%   [fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
%   rho=sqrt(fx.*fx+fy.*fy);
%   f_0=0.4*N;
%   filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
    imageAll=IMAGES(:,:,:,i);  % you will need to provide get_image
    for plane=1:3
        image=imageAll(:,:,plane);
        If=fft2(image);
        imagew=real(ifftshift(ifft2(If./filt)));
        tempIM(:,plane) = reshape(imagew,N^2,1);
    end
    IMAGES(:,:,:,i)=reshape(tempIM,N,N,3,1);
    
end

%   IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));

IMAGES = reshape(IMAGES,N,N,3,M);

figure(2)
clf;
imshow(IMAGES)