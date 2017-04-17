function [model, trainingAcc] = trainHOG(Xtrain, ytrain,sizeFeature, C,iter, tol)


ntrain = size(Xtrain,1);
Xtrain=[Xtrain ones(ntrain,1)]; %so that w.x+b = [x 1].[w;b]
[w alpha]= SVM_DC(Xtrain', ytrain, C, 1, 0,iter,tol);
   ebsilon = 1- ytrain.*(Xtrain * w)';
  

 %prediction
 ypred = double(Xtrain * w > 0);
 for i=1:size(ypred,1)
     if ypred(i)==0
         ypred(i) = -1;
     end
 end
 ypred = ypred';
 
 
 trainingAcc=(1- sum(ytrain ~= ypred)/ntrain);
 fprintf('train Accuracy  %g%%',trainingAcc*100);
 
 model.w= reshape(w(1:end-1),sizeFeature);
 model.b = w(end);
 model.wb = w;
 model.alpha=alpha;
 model.ebsilon = ebsilon;