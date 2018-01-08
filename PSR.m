function y = PSR(r)
a = max(r(:));
b = mean(r(:));
c = std(r(:));
y = (a-b)/c;