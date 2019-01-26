function [I,I_test,lmnI,lmnI_test] = Preparation(imgs_path,imgs_path_test,n)
%Image preparation function for the online demo

files = dir(imgs_path);
j=1;
for i=3:length(files)
   I{j} = imread(strcat(imgs_path,'/',files(i).name));
   I{j} = rgb2gray(I{j});
   I{j} = double(I{j});
   I{j} = I{j}/255;
   sz = size(I{j});
   mid = [round(sz(1)/2),round(sz(2)/2)];
   temp = I{j};
   I{j} = temp(mid(1)-127:mid(1)+128,mid(2)-127:mid(2)+128);
   j=j+1;
end

k = (1/(n^2))*ones(n,n);
lmnI = cell(1,length(I));

for image=1:length(I)
  fprintf('Remove mean: %10d\r',image);
  temp = I{image};
  lmnI{image} = rconv2(temp,k);
  temp = temp - lmnI{image};
  I{image} = temp;
end

%Test images

files = dir(imgs_path_test);
j=1;
for i=3:length(files)
   I_test{j} = imread(strcat(imgs_path_test,'/',files(i).name));
   I_test{j} = rgb2gray(I_test{j});
   I_test{j} = double(I_test{j});
   I_test{j} = I_test{j}/255;
   sz = size(I_test{j});
   mid = [round(sz(1)/2),round(sz(2)/2)];
   temp = I_test{j};
   I_test{j} = temp(mid(1)-127:mid(1)+128,mid(2)-127:mid(2)+128);
   j=j+1;
end

lmnI_test = cell(1,length(I_test));

for image=1:length(I_test)
  fprintf('Remove mean: %10d\r',image);
  temp = I_test{image};
  lmnI_test{image} = rconv2(temp,k);
  temp = temp - lmnI_test{image};
  I_test{image} = temp;
end