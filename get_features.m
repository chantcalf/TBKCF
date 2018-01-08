function x = get_features(im, features, cell_size, cos_window)	
    if features.hog,
		%HOG features, from Piotr's Toolbox
		x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")

    end
	if features.gray,
		%gray-level (scalar feature)
        %x1 = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		%x1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
		xx = double(im) / 255;
		xx = xx - mean(xx(:));
        dx = floor(size(im,1)/cell_size);
        dy = floor(size(im,2)/cell_size);
        for i=1:dx
            for j=1:dy
                x(i,j,:)=reshape(xx((i-1)*cell_size+1:i*cell_size,(j-1)*cell_size+1:j*cell_size),1,cell_size*cell_size);
            end
        end
        %x = cat(3,x,x1);
	end
	
	%process with cosine window if needed
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
