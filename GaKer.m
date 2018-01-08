function k = GaKer(x,y,sigma)
d1 = size(x,2);
d2 = size(y,2);
k = zeros(d1,d2);
for i=1:d1
    for j=1:d2
        k(i,j) = gauss(x(:,i),y(:,j),sigma);
    end
end

function y=gauss(x1,x2,sigma)
y=exp(-1 / sigma^2 * sum((x1-x2).^2));
%y=(sum(x1.*x2))^2;