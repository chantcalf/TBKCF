function s=findnm(r,n,m)
s = zeros(n,m,2);
[a,b] = size(r);
dn = a/n;
dm = b/m;
for i = 1:n
    for j = 1:m
        r1 = r(floor((i-1)*dn)+1:floor(i*dn),floor((j-1)*dm)+1:floor(j*dm));
        [x0,y0] = find(r1 == max(r1(:)),1);
        s(i,j,1)=x0+floor((i-1)*dn);
        s(i,j,2)=y0+floor((j-1)*dm);
    end
end
