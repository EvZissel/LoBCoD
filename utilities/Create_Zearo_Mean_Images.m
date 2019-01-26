function [I] = Create_Zearo_Mean_Images(imgs_path,n)

    files = dir(imgs_path);
    j=1;
    for i=3:length(files)
            I{j} = imread(strcat(imgs_path,'/',files(i).name));
            I{j} = rgb2gray(I{j});
            I{j} = double(I{j});
            j=j+1;
    end
    k = (1/(n^2))*ones(n,n);

    for image=1:length(I)
       fprintf('Remove mean: %10d\r',image);
       temp = I{image};
       lmn = rconv2(temp,k);
       temp = temp - lmn;
       I{image} = temp;
    end
end