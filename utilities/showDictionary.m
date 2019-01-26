function showDictionary(D)

[n,m] = size(D);

clf;
if true
    vl_imarraysc(reshape(D, sqrt(n), sqrt(n), m), 'spacing', 2);
else
    pd = 2;
    sqr_k = ceil(sqrt(m));
    d_disp = zeros(sqr_k * [sqrt(n) + pd, sqrt(n) + pd] + [pd, pd]);
    for j = 0 : m - 1
        d_disp(floor(j/sqr_k) * (sqrt(n) + pd) + pd + (1:sqrt(n)) , mod(j,sqr_k) * (sqrt(n) + pd) + pd + (1:sqrt(n))) = reshape(D(:,j + 1), sqrt(n), sqrt(n)); 
    end
    imagesc(d_disp);
end
colormap gray;
axis equal;
set(gca,'visible','off');
axis image off;

