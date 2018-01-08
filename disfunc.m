function d = disfunc(p,pos1,pos0)
d = zeros(size(p,1),size(p,2));
dx = pos1 - pos0-[1,1];
for i=1:size(p,1)
    for j=1:size(p,2)
        dis = [i,j]+dx;
        d(i,j) = sqrt(sum(dis(:).^2));
    end
end
m = max(d(:));
d = -0.5/m*d+0.5;
d = d.^0.5;
