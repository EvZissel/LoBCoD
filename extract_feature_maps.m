function [alpha,I_rec] = extract_feature_maps(feature_maps,n,m,sz,D)

alpha_mapM = cell(1);
N =size(feature_maps,2);
alpha = cell(size(feature_maps));
I_patches_rec =  cell(size(feature_maps));
I_rec = cell(1,N);


for i=1:N
    for j=1:m
         alpha_mapM{1} = myim2col_set_nonoverlap(feature_maps{i}{j},n);
        for k=1:n^2
            alpha{i}{k}(j,:) = alpha_mapM{1}{k}(1,:);
        end
    end
    for k=1:n^2
        I_patches_rec{i}{k} = D*alpha{i}{k};
    end 
    I_rec{i} = mycol2im_set_nonoverlap(I_patches_rec{i},sz,n);
end


end