function [res,feat] = detect(im,model,maxnum)
% [res,feat] = detect(im,model,maxnum)

[h,w,nf] = size(model.w);

%Pre-rotate so that a convolution is handled accordingly
ww = model.w;
for i = 1:nf,
  ww(:,:,i) = rot90(ww(:,:,i),2);
end
beta = model.b * model.sb;

% Compute feature pyramid
[featpyr,scales] = featpyramid(im,model.sbin,model.interval);
featpyr{1} = features(im, model.sbin);
scales(1)=1;

% Pre-allocate space for features
if nargin > 2
  maxn = 0;
  for s = 1:length(featpyr)
    maxn = maxn + (size(featpyr{s},1)+h-1) * (size(featpyr{s},2)+w-1);
  end
  maxnum = min(maxnum,maxn);
  feat = zeros(model.len,maxnum,'single');
  cnt  = 1;
end

res = [];
for s = 1:length(scales),
  featIm  = featpyr{s};
  
  % Pad
 featIm  = padarray(featIm,[h w]-1,0,'both');
  
  % Score each location
  fsiz = size(featIm(:,:,1));
  resp = zeros(fsiz - [h w] + 1);
  if isempty(resp),
    break;
  end  
  for i = 1:nf,
    resp = resp + conv2(featIm(:,:,i),ww(:,:,i),'valid');
  end

  resp = resp + beta;            
  I = find(resp >prctile(resp(:),99.0));% model.thresh);
  I = I(:);
  num = length(I);
  if num > 0,
    [y,x] = ind2sub(size(resp),I);
    sc = model.sbin*scales(s);
     % Scale accounting for 0-1 indexing: y' = (y-1)*sc + 1
    % Add in 1 cell-shift from image feature computation: y' = (y-1+1)*sc+1 = y*sc+1
    % Recall that [h w] subtracts off a one-cell border
    % eg, model.siz/model.sbin = 1 + [h w] + 1
    % y' = (y-1)*sc+1
    % Finally, account for padded image: y' = (y-1-h+1)*sc+1 = (y-h)*sc+1
    b.y  = (y-h)*sc + 1;
    b.x  = (x-w)*sc + 1;
    b.h  = repmat((1+h+1)*sc,num,1);
    b.w  = repmat((1+w+1)*sc,num,1);
    b.r  = resp(I);
    b.r  = b.r(:);
    b.id = uint16([repmat(s,size(b.r)) x y]);
    res  = add(res,b);
    if nargin > 2,
      %Write out features
      for i = 1:length(I),
        dat = featIm(y(i):y(i)+h-1,x(i):x(i)+w-1,:);
        feat(:,cnt) = [dat(:); model.sb];
        cnt = cnt + 1;
        if cnt > maxnum,
          res = sub(res,1:maxnum);
          return;
        end
      end
    end
  end
end

if nargin > 2,
  feat = feat(:,1:cnt-1);
end

