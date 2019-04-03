function [I] = Load_Text_Images(imgs_path)

    files = dir(imgs_path);
    j=1;
    for i=3:length(files)
            I{j} = im2double(imread(strcat(imgs_path,'/',files(i).name)));
            j=j+1;
    end
end