%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This functino is to whiten an image in the fourier domain.
%
% @file
% @author Matthew Zeiler
% @date Jun 28, 2010
%
% @image_file @copybrief whiten.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
%   N=image_size;
N=100;
%   M=num_images;
M=1;

[fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho=sqrt(fx.*fx+fy.*fy);
f_0=0.4*N;
filt=rho.*exp(-(rho/f_0).^4);

for i=1:M
    imageAll=double(imread('1.jpg'));  % you will need to provide get_image
    for plane=1:3
        image=imageAll(:,:,plane);
        If=fft2(image);
        imagew=real(ifft2(If.*fftshift(filt)));
        tempIM(:,plane) = reshape(imagew,N^2,1);
        %                 tempIM(:,plane)=sqrt(1)*tempIM(:,plane)/sqrt(mean(var(tempIM(:,plane))));
        
    end
    
    IMAGES(:,i)=reshape(tempIM,N^2*3,1);
    
end

IMAGES=sqrt(0.1)*IMAGES/sqrt(mean(var(IMAGES)));

IMAGES = reshape(IMAGES,N,N,3,M);

figure(1)
clf;
% sdispims(IMAGES);
imshow(IMAGES);
unwhiten