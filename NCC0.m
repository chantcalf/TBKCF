function f = NCC0(x,y)
x = x-mean(x(:));
y = y-mean(y(:));
a = x.*y;
b = x.^2;
c = y.^2;
f = sum(a(:))/sqrt(sum(b(:))*sum(c(:)));
