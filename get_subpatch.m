function [out,pos1] = get_subpatch(im, pos, sz)
n = size(im,1);
m = size(im,2);

x0 = floor(pos(1)) + 1 - floor(sz(1)/2);
y0 = floor(pos(2)) + 1 - floor(sz(2)/2);
x1 = floor(pos(1)) + sz(1) - floor(sz(1)/2);
y1 = floor(pos(2)) + sz(2) - floor(sz(2)/2);
if x0<1 x0=1;end
if y0<1 y0=1;end
if x1>n x1=n;end
if y1>m y1=m;end
pos1 = [x0,y0];
out = im(x0:x1,y0:y1,:);