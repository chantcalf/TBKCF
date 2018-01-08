function y = overlap(x1,x2,t)
d = abs(x1-x2);
lap = t-d;
lap(lap<0)=0;
y= lap(1)*lap(2)/(2*t(1)*t(2)-lap(1)*lap(2));