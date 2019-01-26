function patches = myim2col(im, n)

padded = padarray(im, [n-1,n-1], 'both');
patches = im2colstep(padded, [n,n]);
