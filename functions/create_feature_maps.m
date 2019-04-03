function [feature_maps,I_rec] = create_feature_maps(alpha,n,m,sz,D)

alpha_mapM = cell(1);
feature_maps = cell(1,size(alpha,2));
N =size(alpha,2);
one_vec = zeros(n^2,1);
one_vec(1) = 1;
alpha_rec = cell(size(alpha));
I_patches_rec =  cell(size(alpha));
I_rec = cell(1,N);
for i=1:N
    for j=1:m
        for k=1:n^2
            alpha_mapM{1}{k} = one_vec*alpha{i}{k}(j,:);

        end
        feature_maps{i}{j} = mycol2im_set_nonoverlap(alpha_mapM{1},sz,n);
    end

end

for i=1:N
    for j=1:m
         alpha_mapM{1} = myim2col_set_nonoverlap(feature_maps{i}{j},n);
        for k=1:n^2
            alpha_rec{i}{k}(j,:) = alpha_mapM{1}{k}(1,:);
        end
    end
    for k=1:n^2
        I_patches_rec{i}{k} = D*alpha_rec{i}{k};
    end 
    I_rec{i} = mycol2im_set_nonoverlap(I_patches_rec{i},sz,n);
end


end