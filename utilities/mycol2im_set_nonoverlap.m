function im = mycol2im_set_nonoverlap(patches, sz, n)

cnt = 1;
im = zeros(sz + 2*(n-1));
for dx = 1:n
    for dy = 1:n
        cur_sz = sz + 2*(n-1) - [dy-1, dx-1];
        rec = col2imstep(patches{cnt}, cur_sz, [n,n], [n,n]);
        im(dy:end, dx:end) = im(dy:end, dx:end) + rec;
        cnt = cnt + 1;
    end
end

im = im(n:end-n+1, n:end-n+1);

return


