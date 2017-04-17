function showBoxes(im,boxes)
%showBlob(im,boxes)

cla;
imagesc(im);
hold on;

if ~isempty(boxes),
  x1 = boxes.x;
  x2 = boxes.x+boxes.w-1;
  y1 = boxes.y;
  y2 = boxes.y+boxes.h-1;
  line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]','color','r','linewidth',4);
end

drawnow;
hold off;