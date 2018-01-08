function AnsShow(I,y,target,f)
subplot(221);
imshow(I);title(num2str(f));hold on;

x0=target(1);
y0=target(2);
ww=target(3);
h=target(4);
plot([x0,x0+ww,x0+ww,x0,x0],...
    [y0,y0,y0+h,y0+h,y0],'-+r','LineWidth',2);

subplot(222);
imshow(double(y));title(num2str(f));hold on;

im = I;


pause(0.01);
hold off;
