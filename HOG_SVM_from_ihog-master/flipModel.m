function [len,fI] = flipModel(model)
% [len,flipI] = flipModel(model)
% Computes indexes that will flip the weights of a model
% len is the length of the symmetric model 
% (typically half the length of the original model)

siz = size(model.w);
n   = prod(siz);
I   = reshape(1:n,siz);
fI  = flipHOG(I);
len = sum(I(:) <= fI(:)) + 1;
fI  = [fI(:)' model.len];

function f =  flipHOG(w)
% f = flipHOG(w)
% Left/right flips a HOG 3-D tensor
% 4x9 (normalizations X orientations) is hard-coded
%
% Orientation:
% | \ - ....
%
% 1 2 3 4 5 6 7 8 9
% 1 9 8 7 6 5 4 3 2

% Normalizaiton:
% 1 x     x x      x 3     x x
% x x     2 x      x x     x 4
%
% 1 2 3 4
% 3 4 1 2

normI = [3 4 1 2];
orientI = [1 9:-1:2];
f = w(:,end:-1:1,[orientI normI+9]);
