clear;
addpath(genpath('.'))
maxpossamples=30000;
maxnegsamples=30000;
EachHardNeg=60000;ith_hardNeg=1;
normImageNum=255;
dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/pos/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/pos/*_64x128.jpg');
TotalImages=length(listing);
%numSamplesPerImage=800;
%winsize=32;
%totalSamples=TotalImages*numSamplesPerImage;
%Xneg = uint8(zeros(32,32,3,totalSamples));

% Model parameters
model.interval = 1; % 4;   %Number of discrete scales in an octave for image pyramid
model.sbin = 8;        %Size of spatial bin
model.siz  = [128 64]; %Size of model
model.isiz = model.siz - 2*16;    %Size of model w/o background context
model.b  = 0;  % Bias of SVM
model.sb = 10; % Scale factor for model.b
model.thresh = 0;
h  = model.siz(1)/model.sbin - 2;
w  = model.siz(2)/model.sbin - 2;
nf = length(features(zeros([3 3 3]),1));
model.w = zeros([h w nf]);
model.len = prod(size(model.w))+1;

% Prepare model for mirror-flipping
%[len,model.flipI] = flipModel(model);
ids = zeros(4,200000,'uint16');


feat_pos_train = zeros(14,6,32,TotalImages);
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename))/normImageNum;    
    featIm = features(im,model.sbin);
    %feat = featIm(3:end-2,3:end-2,:);
    feat = [featIm(:); model.sb];
    %feat = symFeat(feat,model.flipI);
    Xtrain(:,i) = feat;
    ytrain(i)=1;
    ids(:,i) = [i 0 0 0]; % see def of ids at hardenging part
    
    
    if(i>=maxpossamples)
        break;
    end
  
 
   if 0
    ihog = invertHOG( feat_pos_train(:,:,:,1));
     figure;
     subplot(131);
    imagesc(im); axis image; axis off;
    title('Original Image', 'FontSize', 20);

    subplot(132);
    showHOG(feat_pos_train); axis off;
    title('HOG Features', 'FontSize', 20);

    subplot(133);
    imagesc(ihog); axis image; axis off;
    title('HOG Inverse', 'FontSize', 20);
   end
  

end
ntrainpos = i;
sizeFeature=size(featIm);







dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/neg/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/neg/*.jpg');
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename))/normImageNum;    
    featIm = features(im,model.sbin);
 %   feat = featIm(3:end-2,3:end-2,:);
    feat = [featIm(:); model.sb];
    %feat = symFeat(feat,model.flipI);
    Xtrain(:,i+ntrainpos) = feat;
    ytrain(i+ntrainpos)=-1;
    ids(:,i+ntrainpos) = [i+ntrainpos 0 0 0]; % see def of ids at hardenging part
  if(i>=maxnegsamples)
        break;
    end
end
ntrainneg= i;
posi=[ones(1,ntrainpos) , zeros(1,ntrainneg)   ];


%Optimization%
C=1;iter=1000;tol=0.001;
Xtrainy=repmat(ytrain,size(Xtrain,1),1).*Xtrain;
[w cache]= qpopt( single(Xtrainy) , single(ones(1,size(ytrain,2))), C,tol);
% w = unsymWeight(w0,model.flipI);
 model.w = reshape(w(1:end-1),size(model.w));
 model.b = w(end);



 
 
 % Test Preparation
 
 dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/pos/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/pos/*.jpg');
TotalImages=length(listing);

feat_pos_test = zeros(14,6,32,TotalImages);
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename))/normImageNum;    
    feat_pos_test(:,:,:,i) = features(im,8);
    if(i>=maxpossamples)
        break;
    end
  
end
ntestpos = i;

dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/neg/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/neg/*.jpg');
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename))/normImageNum;    
    feat_neg_test(:,:,:,i) = features(im,8);
  if(i>=maxnegsamples)
        break;
    end
end
ntestneg= i;

Xtest=[];ytest=[];
for  i=1:ntestpos
Xtest(:,i) = reshape(feat_pos_test(:,:,:,i),size(feat_pos_test,1)*size(feat_pos_test,2)*size(feat_pos_test,3),1); 
ytest(i)=1;
end
for i=1:ntestneg
Xtest(:,i+ntestpos) = reshape(feat_neg_test(:,:,:,i),size(feat_neg_test,1)*size(feat_neg_test,2)*size(feat_neg_test,3),1); 
ytest(i+ntestpos)=-1;
end




%-------------- Testing --------------
ntest = size(Xtest,2);
%Xtest2=[Xtest' ones(ntest,1)];
Xtest2 = Xtest';
ypred = double(Xtest2 * model.w(:) > 0);
 for i=1:size(ypred,1)
     if ypred(i)==0
         ypred(i) = -1;
     end
 end
 ypred = ypred';
 
 
 
 testingAcc=(1- sum(ytest ~= ypred)/ntest);
 positiveTestingAcc = 1-(sum((ytest ~= ypred )& (ytest==1)))/ntestpos;
 fprintf('Testing Accuracy  %g%%, positive Testing acc %g%%',testingAcc*100,positiveTestingAcc*100);

 wShow = reshape(model.w,sizeFeature); %model.wb(end) = b
 figure;showHOG(wShow);
 %figure; wShowInv = invertHOG(wShow);  imagesc(wShowInv);
 
 


 
 
 TestImage = double(imread('testImg.jpg'))/normImageNum;
   [blobs,feat] = detect(TestImage,model,100000);  
 res = sub(blobs,~is_boxOutside(blobs,size(TestImage)));
 showBoxes(TestImage,res); drawnow;
 
 save beforHarden_qp20140128;
 
model = hardenTraining(ids,ntrainpos,ntrainneg,model,cache,posi, Xtrain,ytrain,C,tol);
    
 
 TestImage = double(imread('testImg.jpg'))/normImageNum;
   [blobs,feat] = detect(TestImage,model,100000);  
 res = sub(blobs,blobs.r>prctile(blobs.r,70));
 %showBoxes(TestImage,res); drawnow;
 

 
 numDetect = size(res.x,1);
 for i = 1:numDetect
boxx(i,1) = res.x(i);
boxx(i,2) = res.y(i);
boxx(i,3) = res.x(i) + res.w(i);
boxx(i,4) = res.y(i) + res.h(i);
 end
 
topp = nms(boxx,0.5);
res_nms.x = res.x(topp);
res_nms.y = res.y(topp);
res_nms.w = res.w(topp);
res_nms.h = res.h(topp);
res_nms.id = res.id(topp);
res_nms.r = res.r(topp);
 showBoxes(TestImage,res_nms); drawnow;
 
  save afterHarden_qp20140311;
 