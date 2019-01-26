function [I,noisyI,M,noisyImean,lmnI] = Create_Zearo_Mask(imgs_path,n)

% This function creates images with holes and substructs its mean using
% the uncurrapted pixels   
          
    files = dir(imgs_path);
    j=1;
    for i=3:length(files)
            I{j} = imread(strcat(imgs_path,'/',files(i).name));
            %I{j} = rgb2gray(I{j}); %open for colored images
            I{j} = double(I{j});
            j=j+1;
    end

    k2 = (1/(n^2))*ones(n,n);
    k = ones(n,n);
    M = cell(1,length(I));
    noisyI =  cell(1,length(I));
    sz =  cell(1,length(I));
    noisyImean = cell(1,length(I));
    lmnI = cell(1,length(I));
    for image=1:length(I)
        sz{image} = size(I{image}); 
        rng(0);
        M{image} = ones(sz{image});
        M{image}(rand(sz{image}) < 0.50) = 0; %masks 50% of the pixels
        noisyI{image} = M{image}.*I{image};
    

       fprintf('Remove mean: %10d\r',image);
       temp = noisyI{image};
       lmn = rconv2(temp,k);
       imn_size = rconv2(M{image},k);
       noisyImean{image} = lmn./imn_size;
       temp = M{image}.*(temp - noisyImean{image});
       noisyI{image} = temp;
       
       tempI = I{image};
       lmnI{image} = rconv2(tempI,k2);
       tempI = tempI - lmnI{image};
       I{image} = tempI;
    end
end