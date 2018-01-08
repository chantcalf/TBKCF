function bx = get_context(im,pos,window_sz,target_sz)
n = size(im,1);
m = size(im,2);
[patcht,pt] = get_subpatch(im, pos, target_sz);
[patchc,pc] = get_subpatch(im, pos, window_sz);
n1 = size(patchc,1);
m1 = size(patchc,2);
n2 = size(patcht,1);
m2 = size(patcht,2);

bx = double(patchc);
bx_ave = (sum(patchc(:))-sum(patcht(:)))/(n1*m1-n2*m2);
bx = bx-bx_ave;
xx1 = pt(1)-pc(1);
yy1 = pt(2)-pc(2);
bx(xx1+1:xx1+n2,yy1+1:yy1+m2,:)=0;

%{
x0 = floor(pos(1)) + 1 - floor(window_sz(1)/2);
y0 = floor(pos(2)) + 1 - floor(window_sz(2)/2);
x1 = floor(pos(1)) + window_sz(1) - floor(window_sz(1)/2);
y1 = floor(pos(2)) + window_sz(2) - floor(window_sz(2)/2);
if x0<1 x0=1;end
if y0<1 y0=1;end
if x1>n x1=n;end
if y1>m y1=m;end
pos1 = pos - [x0,y0] + [1,1];
out0 = double(im(x0:x1,y0:y1,:));
x2 = floor(pos(1)-target_sz(1)/2) + 1;
y2 = floor(pos(2)-target_sz(2)/2) + 1;

t0 = im(x2:x2+target_sz(1)-1,y2:y2+target_sz(2)-1,:);

n1 = x1-x0+1;
m1 = y1-y0+1;
n2 = target_sz(1);
m2 = target_sz(2);


out = out0(:,:,i);
t = t0(:,:,i);
bx_ave = (sum(double(out(:)))-sum(t(:)))/(n1*m1-n2*m2);
bx(:,:,i) = out-bx_ave;

xx1 = x2-x0;
yy1 = y2-y0;
bx(xx1+1:xx1+n2,yy1+1:yy1+m2,:)=0;
%}