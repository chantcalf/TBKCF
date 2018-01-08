function labels = TBlabel(window_sz,cell_size,padding)
sz = floor(window_sz/cell_size);
labels = zeros(sz);
dd = padding/2.0;

target = floor([dd*window_sz(1),(1+dd)*window_sz(1),dd*window_sz(2),(1+dd)*window_sz(2)]/(1+padding));
for i = 1:sz(1)
    for j = 1:sz(2)
        labels(i,j) = overlap(target,[i-1,i,j-1,j]*cell_size+[1,0,1,0]);
    end
end


function y = overlap(a,b)
x0=max(a(1),b(1));
x1=min(a(2),b(2));
y0=max(a(3),b(3));
y1=min(a(4),b(4));
area=0;
if (x0<x1 && y0<y1)
    area=(x1-x0+1)*(y1-y0+1);
end
y=area/16;
%{
if y>0.5
    y=1;
else
    y=0;
end
%}