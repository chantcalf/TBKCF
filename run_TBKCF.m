function results=run_TBKCF(seq, res_path, bSaveImage)


close all

feature_type = 'hog';
kernel_type = 'gaussian';
kernel.type = kernel_type;
	
features.gray = false;
features.hog = false;

padding = 1.5;  %extra area surrounding the target
lambda = 1e-4;  %regularization
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

switch feature_type
	case 'gray',
		interp_factor = 0.075;  %linear interpolation factor for adaptation

		kernel.sigma = 0.2;  %gaussian kernel bandwidth
		
		kernel.poly_a = 1;  %polynomial kernel additive term
		kernel.poly_b = 7;  %polynomial kernel exponent
	
		features.gray = true;
		cell_size = 4;
		
	case 'hog',
		interp_factor = 0.02;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
    otherwise
    error('Unknown feature.')
end

itx = 0.08;
video_path = '';
img_files = seq.s_frames;
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);
resize_times = 1;
check_size = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
check_size1 = target_sz(1)<=50 && target_sz(2)<=50;
if check_size ==1
    resize_times = 0.5;
end
if check_size1 ==1
    resize_times = 2;
end
resize_image = check_size || check_size1;
if resize_image,
    pos = floor(pos * resize_times);
    target_sz = floor(target_sz * resize_times);
end


window_sz = floor(target_sz * (1 + padding)); %kcf window
window_sz2 = floor(target_sz * (1 + 3.5)); %short-term tracking window
I = imread([video_path img_files{1}]);
if size(I,3) > 1,
			I = rgb2gray(I);
end
rsz = floor(size(I)*0.5);
if rsz(1)<window_sz2(1)
    rsz(1)=window_sz2(1);
end
if rsz(2)<window_sz2(2)
    rsz(2)=window_sz2(2);
end


alapha = sqrt(sum(window_sz2(:).^2))/4;     
[rs, cs] = ndgrid((1:window_sz2(1)) - floor(window_sz2(1)/2), (1:window_sz2(2)) - floor(window_sz2(2)/2));
dist = rs.^2 + cs.^2;
conf0 = exp(-0.5 / (alapha) * sqrt(dist));

ssz = floor(target_sz/cell_size/2);   
sz_t = 1;
target_sz2 = floor(target_sz * sz_t);
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
time = 0; 
v0 = [0,0];

trackerror = 0;	
is_reliable = 1;

pp6 = 0;

p1res = zeros(numel(img_files),1);
Need_Redetection = 1;  %open redetection
ta = 0.35;
tb = 0.25;
tr = 0.33;
poor = 0;
for frame = 1:numel(img_files),
        %frame
		I = imread([video_path img_files{frame}]);
        
        if size(I,3) > 1,
			I = rgb2gray(I);
        end
        
		if resize_image,
			I = imresize(I, resize_times);
        end
        im = I;
        
        %maxN = size(I,1);
        %maxM = size(I,2);
        
        pos0 = pos;
        red = 0;
		
        tic()
        if frame > 1, 
            %start short-term tracking
            NCCwork = 0;
            [patchB,posPB] = get_subpatch(im, pos, window_sz2);
            rpos = posPB - pos + floor(window_sz2/2)-[1,1];
            patch_sz = size(patchB);
            conf = conf0(rpos(1)+1:rpos(1)+patch_sz(1),rpos(2)+1:rpos(2)+patch_sz(2));
            
            %context tracking
            responseB = NCCB(double(patchB),bx,size(model_tx));
            responseB = responseB.*conf;
            
            [x0,y0] = find(responseB == max(responseB(:)),1);
            xc = posPB+[x0,y0]-[1,1];
            
            dd1 = max(responseB(:));

            %target tracking
            responseNB = NCC(double(patchB),model_tx);
            responseNB = responseNB.*conf;
          
            %[x0,y0] = find(responseNB == max(responseNB(:)),1);
            %xN = posPB+[x0,y0]-[1,1];
            dd2 = max(responseNB(:));
            
            %integration
            responseB =cycm(responseB,floor(v0)); 
            responseB1 =(responseB*dd1 + responseNB*dd2)/(dd2+dd1);  
            responseB1(responseNB<dd2/2)=0;
            if poor
                responseB1 = responseNB;
            end
            [x0,y0] = find(responseB1 == max(responseB1(:)),1);
            pb = max(responseB1(:));

            xm = posPB + [x0,y0]-[1,1];
            
            %check the rough tracking result
            if pb>0.35 && responseNB(x0,y0)>0.3
                pos = xm;
                NCCwork = 1;
            else
                xm = pos0;
            end
            
            patch = get_subwindow(im, pos, window_sz);
            feat = get_features(patch, features, cell_size, cos_window);
            [zf,response0] = detection(feat, kernel,model_xf,model_alphaf);
            p=max(response0(:));
            
            %check the kcf results in the rough predicted position and in the frame 
            if NCCwork == 1
                patch0 = get_subwindow(im, pos0, window_sz);
                feat0 = get_features(patch0, features, cell_size, cos_window);
                [zf,response01] = detection(feat0, kernel,model_xf,model_alphaf);
                p01=max(response01(:));
                if p01>p
                    NCCwork = 0;
                    p = p01;
                    response0 = response01;
                    patch = patch0;
                    pos = pos0;
                end
     
            end
            if p<0.17
                poor = 1;
            else
                poor = 0;
            end
            [vert_delta, horiz_delta] = find(response0 == max(response0(:)), 1);
            if vert_delta > size(zf,1) / 2,  
                vert_delta = vert_delta - size(zf,1);
            end
            if horiz_delta > size(zf,2) / 2,  
                horiz_delta = horiz_delta - size(zf,2);
            end
            pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            pos = check(pos,target_sz2,size(im)); 
            
            v0 = pos-xc;
            if poor || NCCwork==0
                v0 = [0,0];
            end
            
            %calculate reliable kcf result
            patchc = get_subwindow(im, pos, window_sz);
            featc = get_features(patchc, features, cell_size, cos_window);
            [zf,response1] = detection(featc, kernel,re_model_xf,re_model_alphaf);
            pp=response1(1,1);

            %redetection
            if Need_Redetection
                
                if frame==6     %Re-detection doesn't start in the videos with bad tracking result in frame 6
                    pp6 = pp;
                    if pp6<0.25
                        Need_Redetection = 0;
                    elseif pp6<0.35
                        rx = pp6;
                    end
                end
                
                if Need_Redetection
                    first_error = 0;
                    is_reliable = frame<6 || pp>=ta; %check reliability
                    if (is_reliable) %need not redetect when the result is reliable
                        trackerror = 0;
                    elseif ( ~trackerror && pp<tb) %check tracking error
                        %if error then save paras before error
                        trackerror = 1;
                        model_alphaf1 = model_alphaf;
                        model_xf1 = model_xf;
                        model_tx1 = model_tx;
                        posre = pos;
                        first_error = 1;
                        red_t = frame;
                    end
                    %[is_reliable,trackerror]
                    if (trackerror && (first_error || mod(frame,3)==1)) %redetect the target
                        red = 1;                        
                        %patch1 = im;
                        %pos1 = [1,1];
                        [patch1,pos1] = get_subpatch(im,posre,rsz);
                        responseN = NCC(double(patch1),re_model_tx);
                        %{
                        if first_error
                            patchf = patch1;
                            responsef = responseN;
                        end
                        %}
                        ca = findnm(responseN,5,5);
                        p1 = 0;
                        xx0 = 0;
                        yy0 = 0;
                        %subgrid
                        for i=1:5
                            for j=1:5
                                x0 = ca(i,j,1);
                                y0 = ca(i,j,2);
                                if responseN(x0,y0)>0.3 
                                    patchx = get_subwindow(patch1, [x0,y0], window_sz);
                                    featx = get_features(patchx, features, cell_size, cos_window);
                                    [zf,responsex] = detection(featx, kernel,re_model_xf,re_model_alphaf);
                                    p2=max(responsex(:));
                                    %p2=responsex(1,1);
                                    if p2>p1
                                        
                                        [vd, hd] = find(responsex == max(responsex(:)), 1);
                                        if vd > size(zf,1) / 2,  
                                            vd = vd - size(zf,1);
                                        end
                                        if hd > size(zf,2) / 2,  
                                            hd = hd - size(zf,2);
                                        end
                                        xx0 = x0 + cell_size * (vd - 1);
                                        yy0 = y0 + cell_size * (hd - 1);

                                        p1 = p2;   
                                    end
                                end
                            end
                        end
                        p1res(frame,:) = p1;
                        if first_error  %record the target's missing position
                            fep = p1;
                        else 
                            is_reliable = (p1>=tr && p1>fep*1.1); %check redetection result
                            if is_reliable %recover tracking %PS: here is the key part in redetection, other parts are tricks for performance
                                    pos = pos1+[xx0,yy0]-[1,1];
                                    pos = check(pos,target_sz2,size(im));
                                    v0 = [0,0];
                                    %viewchange = [0,0];
                                    trackerror = 0;        
                                    model_alphaf = model_alphaf1; %recover model paras
                                    model_xf = model_xf1;
                                    model_tx = model_tx1;
                             else
                                if ~poor && frame - red_t > 100 %recover tracking after failure redection for 100 frames and when the short result is not poor
                                    is_reliable = 1;
                                    re_model_alphaf = re_model_alphaf *0.5 + model_alphaf*0.5;
                                    re_model_xf = re_model_xf  *0.5 + model_xf*0.5;
                                    re_model_tx = re_model_tx *0.5 + model_tx*0.5;
                                end
                            end
                        end
                   end
                end
            end
        end
        
        %update context
        bx = get_context(im,pos,window_sz,target_sz2);

        %update target and kcf models
            patch0 = get_subwindow(im, pos, window_sz);
            feat = get_features(patch0, features, cell_size, cos_window);
            [xf,alphaf] = train(feat,kernel,yf,lambda);
            tx = get_target(im,pos,target_sz2);

            if frame == 1,  %first frame, train with a single image
                model_alphaf = alphaf;
                model_xf = xf;
                model_tx = tx;
                re_model_alphaf = alphaf;
                re_model_xf = xf;
                re_model_tx = tx;
            end

            if frame>1,
                if (~poor)
                    model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
                    model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
                    model_tx = (1-itx) * model_tx + itx * tx;
                end

                if (Need_Redetection && is_reliable)
                    re_model_alphaf = (1 - interp_factor) * re_model_alphaf + interp_factor * alphaf;
                    re_model_xf = (1 - interp_factor) * re_model_xf + interp_factor * xf;
                    re_model_tx = (1-itx) * re_model_tx + itx * tx;
                end           
            end


        rect = [pos-floor(target_sz/2),target_sz];
        rect = [rect(2),rect(1),rect(4),rect(3)];
        res(frame,:) = rect;
		time = time + toc();

        
        if frame>1
            
            rr = response0;
            [xx,yy]=size(rr);
            for i=1:xx
                for j=1:yy
                    rr(mod(i+floor(xx/2)-1, xx) + 1,mod(j+floor(yy/2)-1,yy)+1)=response0(i,j);
                end
            end
            subplot(251);imshow(rr);title(frame);
            subplot(256);imshow(patch);
            hold on;
            [x0,y0]=find(rr == max(rr(:)), 1);
            rectangle('Position',[y0-ssz(2),x0-ssz(1),ssz(2)*2,ssz(1)*2]*cell_size,...
                     'Linewidth',2,'LineStyle','-','edgecolor','r');
            hold off;
            

            rr = responseB1;
            subplot(254);imshow(rr);
            subplot(259);imshow(patchB);

            hold on;
            [x0,y0]=find(rr == max(rr(:)), 1);
            
            
            rectangle('Position',[y0-floor(target_sz(2)/2),x0-floor(target_sz(1)/2),target_sz(2),target_sz(1)],...
                     'Linewidth',2,'LineStyle','-','edgecolor','r');
            if (red==1)

                rr = responseN;
                subplot(252);imshow(rr);
                subplot(257);imshow(patch1);
                hold on;
              
                x0 = xx0;
                y0 = yy0;

                rectangle('Position',[y0-floor(target_sz(2)/2),x0-floor(target_sz(1)/2),target_sz(2),target_sz(1)],...
                         'Linewidth',2,'LineStyle','-','edgecolor','r');
                hold off;

            end

            pause(0.001);
            
        end
end

    if resize_image,
        res = floor(res / resize_times);
    end
    
fps = numel(img_files) / time;

disp(['fps: ' num2str(fps)])

results.type = 'rect';
results.res = res;%each row is a rectangle
results.fps = fps;

%kcf detection
function [zf,response] = detection(feat, kernel,model_xf,model_alphaf)
zf = fft2(feat);

switch kernel.type
case 'gaussian',
    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
case 'polynomial',
    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
case 'linear',
    kzf = linear_correlation(zf, model_xf);
end

response = real(ifft2(model_alphaf .* kzf)); 

%kcf training
function [xf,alphaf] = train(feat,kernel,yf,lambda)
xf = fft2(feat);
switch kernel.type
case 'gaussian',
    kf = gaussian_correlation(xf, xf, kernel.sigma);
case 'polynomial',
    kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
case 'linear',
    kf = linear_correlation(xf, xf);
end
alphaf = yf ./ (kf + lambda); 

%NCC the target
function r0 = NCC(patch,tx)
r0 = zeros(size(patch));
ae = eps;
[dn,dm] =size(tx);
[n1,m1] = size(patch);
dn = floor(dn/2);
dm = floor(dm/2);
p = double(patch);
p2 = p.^2;
m = ones(size(tx));
sp = fcf(m,p);
sp2 = fcf(m,p2);
r2 = sum(tx(:).^2);
r3 = sqrt((sp2-(sp.^2)/size(tx,1)/size(tx,2)).*r2);
%for i=1:2
    %tx = rot90(tx,2);
r1 = fcf(tx,p);
r = r1./sqrt((sp2-(sp.^2)/size(tx,1)/size(tx,2)+ae).*(r2+ae));
r(r3==0)=0; 
r0 = r0+r;
%end
%r0 = r0/4;
r0(1:dn-1,:)=0;
r0(n1-dn+1:n1,:)=0;
r0(:,1:dm-1)=0;
r0(:,m1-dm+1:m1)=0;

%NCC the context
function r = NCCB(patch,tx,sz)
ae = eps;
[dn,dm] =size(tx);
m = ones(dn,dm);
D = dn*dm - sz(1)*sz(2);
xx1 = floor((dn-sz(1))/2);
yy1 = floor((dm-sz(2))/2);
m(xx1+1:xx1+sz(1),yy1+1:yy1+sz(2))=0;
[n1,m1] = size(patch);
dn = floor(dn/3);
dm = floor(dm/3);
p = double(patch);
p2 = p.^2;
r1 = fcf(tx,p);
sp = fcf(m,p);
sp2 = fcf(m,p2);
r2 = sum(tx(:).^2);
r = r1./sqrt((sp2-(sp.^2)/D+ae).*(r2+ae));
%{
r(1:dn-1,:)=0.5*r(1:dn-1,:);
r(n1-dn+1:n1,:)=0.5*r(n1-dn+1:n1,:);
r(:,1:dm-1)=0.5*r(:,1:dm-1);
r(:,m1-dm+1:m1)=0.5*r(:,m1-dm+1:m1);
%}

r(1:dn-1,:)=0;
r(n1-dn+1:n1,:)=0;
r(:,1:dm-1)=0;
r(:,m1-dm+1:m1)=0;

%convolution 
function c = fcf(a,b)
d = rot90(a,2);
[an,am] = size(a);
[n,m]=size(b);
nn = floor(an/2);
mm = floor(am/2);
sn = n+nn+nn;
sm = m+mm+mm;
d(sn,sm)=0;
b(sn,sm)=0;
af = fft2(d);
bf = fft2(b);
cd = ifft2(af.*bf);
c = cd(nn+1:nn+n,mm+1:mm+m);

%check whether the position is out of the image and revise
function a=pan(pos,sz,target_sz)
a = 1;
if pos(1)<target_sz(1)/2 || pos(1)>sz(1)-target_sz(1)/2
    a=0;
    return;
end
if pos(2)<target_sz(2)/2 || pos(2)>sz(2)-target_sz(2)/2
    a=0;
    return;
end

%circulate shift
function y = cycm(x,v0)
[n,m] = size(x);
y= zeros(n,m);
for i=1:n
    for j=1:m
        ii=i-v0(1);
        jj=j-v0(2);
        if (ii>0 && ii<=n && jj>0 && jj<=m)
            y(i,j)=x(ii,jj);
        end
    end
end

