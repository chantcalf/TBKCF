function x = check(x,sz,sz1)
x0 = floor(x(1)-sz(1)/2) + 1;
y0 = floor(x(2)-sz(2)/2) + 1;
x1 = floor(x(1)-sz(1)/2) + sz(1);
y1 = floor(x(2)-sz(2)/2) + sz(2);

if x0<1
    x(1) = x(1)-x0+1;
end
if y0<1
    x(2) = x(2)-y0+1;
end
if x1>sz1(1)
    x(1) = x(1)-x1+sz1(1);
end
if y1>sz1(2)
    x(2) = x(2)-y1+sz1(2);
end