function model = hardenTraining(ids,ntrainpos,ntrainneg,model,cache,posi, Xtrain,ytrain,C,tol)
% Hardening with Negative Examples
 
 %0. Each postive and negative sample must have image id. Its ids memory is
 %[image_id 0 0 0].  ids keep tracking all sample previously optimized.
 %This includes all positive samples + prev support Vectors ids has the
 %structure  [image_id  scale  xpos  ypos], where xpos and ypos are raw
 %position before calculate back to real x,y position on the detecting
 %image
 
 
 % 1. Detect a Negative Test image, all detections are tracked by idt
 % memory, which has the same structure [image_id scale xpos ypos] with
 % ids. Do the logical intersection to find I, the featured which are non
 % dumplicated
 
 
 
 %2 From Detected Feature, refine them with the I, and add these features
 %to the previously optimzed feature.
 
  
 %3. Retrain SVM to get better model, Do the FPPW measurement and goto 1.
 %Until there are no more negative image.
 %     3.1  update to model only epsilon-optimal w.
 
 %4. Redo 1,2,3 until maximum round reached.
 
 max_harden =10;
 %SVseq = ((model.alpha > 0.1)+(model.alpha<-0.1))>0;
 %XSVtrain=Xtrain(:,SVseq);
 %ySVtrain = ytrain(SVseq);
 %sizeSv= size(Xhardtrain,2);
 dirname = '/working/peerajak/Dataset/INRIAPerson/Test/neg/';
 listing = dir('/working/peerajak/Dataset/INRIAPerson/Test/neg/*.jpg');
n=ntrainpos+ntrainneg;
icnt = 1; % Number of images encountered with a fixed model
iall = 0; % Total number of iterations encountered
obj = 0; %objective function value ||w||^2
lb = 0;
%ipos = 1;
 SHOWING=1;
 fr=0;
 for ii=1:max_harden
     sizeFP=0;
      maxnum = size(Xtrain,2) - n;
     for i = 1: length(listing)
        filename = strcat(dirname,listing(i).name);
        fr = fr+1;
        im =  double(imread(filename))/255;   
        size(cache.sv)
        n
        J=1:n;
         % Prune cache, keeping all positive features
        I = find(cache.sv | posi(J));
        n = length(I);
        J = 1:n;
        Xtrain(:,J) = Xtrain(:,I);
        ytrain(:,J) = ytrain(:,I);
        ids(:,J) = ids(:,I);
 
        disp('hardinging');
        
        [blobs,feat] = detect(im,model,5000);  
%         notOutsideImgSeq = ~is_boxOutside(blobs,size(im));
%        res = sub(blobs,notOutsideImgSeq);
%        feat = dfeat(:,notOutsideImgSeq);     
         if   ~isempty(blobs)             
            if SHOWING ==1
                 res = sub(blobs,blobs.r > -1);
                 showBoxes(im,res); drawnow;
            end
            idt = [repmat(uint16(fr),size(blobs.x)) blobs.id]';
            [dummy,inds] = intersect(idt',ids','rows');
            inds
            I = logical(ones(size(blobs.x)));
            I(inds) = 0;  
             if any(I),   
                r = blobs.r(I);
               obj = obj + C*sum(max(1+r,0));
             % Total error is unbounded if we didn't collect all violations
            if size(feat,2) == maxnum,
              obj = inf;
            end
             J    = (n+1):(n+sum(I));
            
           %  feat = symFeat(feat(:,I),model.flipI);
             Xtrain(:,J)=feat;
             ytrain(J) = -1;
             ids(:,J) = idt(:,I);
             n = J(end);

            end
         end
        disp([i ii n]);
     
    
     Xtrainy=repmat(ytrain(1:n),size(Xtrain(:,1:n),1),1).*Xtrain(:,1:n);
     [w cache]= qpopt( single(Xtrainy) , single(ones(1,n)), C,tol);
     
       if 1 - lb/obj > tol,
        %  w = unsymWeight(w0,model.flipI);
          model.w = reshape(w(1:end-1),size(model.w));
          model.b = w(end);
          iall = iall + icnt;
          icnt = 0;
          obj  = lb;
       end
      icnt = icnt + 1;
    %  ipos = ipos + 1;
     
    
     end
    
      %-------------- Testing --------------

%     ypred = double(Xtest2 *  model.w(:) > 0);
%      for i=1:size(ypred,1)
%          if ypred(i)==0
%              ypred(i) = -1;
%          end
%      end
%      ypred = ypred';
%  
%  
%      testingAcc=(1- sum(ytest ~= ypred)/ntest);
%      positiveTestingAcc = 1-(sum((ytest ~= ypred )& (ytest==1)))/ntestpos;
%      printf('Testing Accuracy  %g%%, positive Testing acc %g%%',testingAcc*100,positiveTestingAcc*100);

 %wShow = reshape(model.w,sizeFeature); %model.wb(end) = b
% figure;showHOG(wShow);
 %figure; wShowInv = invertHOG(wShow);  imagesc(wShowInv);

 end
 fprintf('finished Optimization');
 
     
