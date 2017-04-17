clear;
addpath(genpath('.'))
maxpossamples=30000;
maxnegsamples=60000;
EachHardNeg=60000;ith_hardNeg=1;
dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/pos/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/pos/*_64x128.jpg');
TotalImages=length(listing);
%numSamplesPerImage=800;
%winsize=32;
%totalSamples=TotalImages*numSamplesPerImage;
%Xneg = uint8(zeros(32,32,3,totalSamples));
feat_pos_train = zeros(14,6,32,TotalImages);
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename)/255);    
    feat_pos_train(:,:,:,i) = features(im,8);
    if(i>=maxpossamples)
        break;
    end
   ihog = invertHOG( feat_pos_train(:,:,:,1));
 
   if i==1
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

dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/neg/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/train/neg/*.jpg');
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename)/255);    
    feat_neg_train(:,:,:,i) = features(im,8);
  if(i>=maxnegsamples)
        break;
    end
end
ntrainneg= i;
sizeFeature=size(feat_neg_train(:,:,:,1));
Xtrain=[];ytrain=[];
for  i=1:ntrainpos
Xtrain(:,i) = reshape(feat_pos_train(:,:,:,i),size(feat_pos_train,1)*size(feat_pos_train,2)*size(feat_pos_train,3),1); 
ytrain(i)=1;
end
for i=1:ntrainneg;
Xtrain(:,i+ntrainpos) = reshape(feat_neg_train(:,:,:,i),size(feat_neg_train,1)*size(feat_neg_train,2)*size(feat_neg_train,3),1); 
ytrain(i+ntrainpos)=-1;
end
ith_hardNeg=ith_hardNeg+1;
C=1;iter=1000;tol=0.001;
 [model, trainingAcc] = trainHOG(Xtrain', ytrain,sizeFeature, C,iter, tol);
 model.sbin=8;
 model.sb = 10; %Hyper param for scaling bias
 model.interval=1;
 model.thresh =0;
 model.len = sizeFeature(1)*sizeFeature(2)*sizeFeature(3)+1;

 
 
 % Test Preparation
 
 dirname = '/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/pos/';
listing = dir('/working/peerajak/ChulaQE/Semister5/Implementation/2_Hog_SVM/HOG_SVM_from_ihog-master/test/pos/*.jpg');
TotalImages=length(listing);

feat_pos_test = zeros(14,6,32,TotalImages);
for i = 1:length(listing)
    
    filename = strcat(dirname,listing(i).name);
    im = double(imread(filename)/255);    
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
    im = double(imread(filename)/255);    
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
Xtest2=[Xtest' ones(ntest,1)];
ypred = double(Xtest2 * model.wb > 0);
 for i=1:size(ypred,1)
     if ypred(i)==0
         ypred(i) = -1;
     end
 end
 ypred = ypred';
 
 
 testingAcc=(1- sum(ytest ~= ypred)/ntest);
 printf('Testing Accuracy  %g%%',testingAcc*100);
 wShow = reshape(model.w,sizeFeature); %model.wb(end) = b
 figure;showHOG(wShow);
 %figure; wShowInv = invertHOG(wShow);  imagesc(wShowInv);
 
 
%  ---------------Hardening with Negative Examples and False Positive------------
%  SVseq = ((model.alpha > 0.1)+(model.alpha<-0.1))>0;
%  FPseq = SVseq + (ytrain==-1)
%  
%  while 1
%          for i=EachHardNeg*(ith_hardNeg-1)+1:EachHardNeg*ith_hardNeg
%             Xtrain(:,i+ntrainpos) = reshape(feat_neg_train(:,:,:,i),size(feat_neg_train,1)*size(feat_neg_train,2)*size(feat_neg_train,3),1); 
%             ytrain(i+ntrainpos)=-1;
%         end
%      
%      [model, trainingAcc] = trainHOG(Xtrain', ytrain, sizeFeature, C,iter, tol);
% 
%      %-------------- Testing --------------
%     ntest = size(Xtest,2);
%     Xtest2=[Xtest' ones(ntest,1)];
%     ypred = double(Xtest2 * model.wb > 0);
%      for i=1:size(ypred,1)
%          if ypred(i)==0
%              ypred(i) = -1;
%          end
%      end
%      ypred = ypred';
% 
% 
%       testingAcc=(1- sum(ytest ~= ypred)/ntest);
%      printf('Testing Accuracy  %g%%',testingAcc*100);
%      wShow = reshape(model.w,sizeFeature); %model.wb(end) = b
%      figure;showHOG(wShow);
%      
%         ith_hardNeg=ith_hardNeg+1;
%         if(ith_hardNeg*EachHardNeg >= ntrainneg)
%             break;
%         end
%      
%  end
%  
 

 
 
 TestImage = double(imread('testImg.jpg'))/255;
   [blobs,feat] = detect(TestImage,model,100000);  
 res = sub(blobs,~is_boxOutside(blobs,size(TestImage)));
 showBoxes(TestImage,res); drawnow;
 
 save beforHarden20131113;
 
 % Hardening with Negative Examples
 
 % 1. Remember Support Vectors from Last model, use them as initial positives and
 % negatives training samples
 
 
 %2 Use the trained model to detect Large Negative Image, any detection is
 %therefore False Positive,memorize that HOG  of that False Positive
 %windows and andd them negative samples
 
 
 
 %3. Retrain SVM to get better model, Do the FPPW measurement and goto 1.
 %again k times.
 
 max_harden =2;
 SVseq = ((model.alpha > 0.1)+(model.alpha<-0.1))>0;
 XSVtrain=Xtrain(:,SVseq);
 ySVtrain = ytrain(SVseq);
 %sizeSv= size(Xhardtrain,2);
 dirname = '/working/peerajak/Dataset/INRIAPerson/Test/neg/';
 listing = dir('/working/peerajak/Dataset/INRIAPerson/Test/neg/*.jpg');

 SHOWING=1;
 for ii=1:max_harden
     sizeFP=0;
     Xhardtrain =[]; yhardtrain=[];
     for i = 1: length(listing)
        filename = strcat(dirname,listing(i).name);
        im =  double(imread(filename))/255;    
        [blobs,feat] = detect(im,model,5000);  
        notOutsideImgSeq = ~is_boxOutside(blobs,size(TestImage));
       res = sub(blobs,notOutsideImgSeq);
       feat = feat(:,notOutsideImgSeq);
         if   ~isempty(blobs)
            if SHOWING ==1
                 showBoxes(im,res); drawnow;
            end
            
            for j=1:size(feat,2)                
                sizeFP = sizeFP+1;    
             %   Im_crop_j = imcrop(im,[blobs.x(j,:) blobs.y(j,:)  blobs.w(j,:)  blobs.h(j,:) ]);
                
                Xhardtrain(:,sizeFP)=feat(1:end-1,j);
                yhardtrain(sizeFP) = -1;
            end
            
         end
        disp([i ii sizeFP]);
     end
    
     XHtrain = [XSVtrain, Xhardtrain];
     yHtrain = [ySVtrain, yhardtrain];
      fprintf('optimizing %d samples',size(XHtrain,2));
     [newmodel, trainingAcc] = trainHOG(XHtrain', yHtrain,sizeFeature, C,iter, tol);
     newmodel.sbin=8;
     newmodel.sb = 10; %Hyper param for scaling bias
     newmodel.interval=1;
     newmodel.thresh =0;
     newmodel.len = sizeFeature(1)*sizeFeature(2)*sizeFeature(3)+1;


     %-------------- Testing --------------

ypred = double(Xtest2 * newmodel.wb > 0);
 for i=1:size(ypred,1)
     if ypred(i)==0
         ypred(i) = -1;
     end
 end
 ypred = ypred';
 
 
 testingAcc=(1- sum(ytest ~= ypred)/ntest);
 printf('Testing Accuracy  %g%%',testingAcc*100);
 model = newmodel;
 %wShow = reshape(model.w,sizeFeature); %model.wb(end) = b
% figure;showHOG(wShow);
 %figure; wShowInv = invertHOG(wShow);  imagesc(wShowInv);

 end
 
 
 TestImage = double(imread('testImg.jpg'))/255;
   [blobs,feat] = detect(TestImage,model,100000);  
 res = sub(blobs,~is_boxOutside(blobs,size(TestImage)));
 showBoxes(TestImage,res); drawnow;
 
 save afterHarden20131113;
 
 
 
 