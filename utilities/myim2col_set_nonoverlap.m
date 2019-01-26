function patches = myim2col_set_nonoverlap(im, n)

padded = padarray(im, [n-1,n-1], 'both');

cnt = 1;
patches = cell(1,n^2);
for dx = 1:n
    for dy = 1:n
        tmp = padded(dy:end,dx:end);
        patches{cnt} = im2colstep(tmp, [n,n], [n,n]);
        cnt = cnt + 1;
    end
end

return

