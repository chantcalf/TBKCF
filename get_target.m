function tx = get_target(im,pos,target_sz)
patcht = get_subwindow(im, pos, target_sz);
pat = double(patcht);
meanx = mean(pat(:));
tx = pat-meanx;  
