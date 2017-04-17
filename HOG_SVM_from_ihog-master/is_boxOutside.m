function ArrayTrueFalse=is_boxOutside(box,sizeIm)
ArrayTrueFalse=[];
if ~isempty(box)
ArrayTrueFalse = ((box.x<0 )  +  (box.y<0) + (box.x+box.w > sizeIm(2))  + ( box.y+box.h >sizeIm(1))  > 0); %or operation
end


end
