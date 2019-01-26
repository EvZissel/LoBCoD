function im = mycol2im(patches, sz, n)

summed = col2imstep(patches, sz + 2*(n-1), [n,n], [1,1]);
im = summed(n:end-n+1, n:end-n+1);
