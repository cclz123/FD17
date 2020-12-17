%%
%Step 0: Preparations
function main(Mode, Type)
%run('.\VLF\vlfeat-0.9.16\toolbox\vl_setup');%Setup SLIC Toolkit
%clear all;
clc;

%%
%Step 1: Parameter Initialization
%Mode ='bear';
%Type ='jpg';
path = ['.\sequences\' Mode '\'];
s=strcat('mkdir ',['.\result\' Mode '\ColorSaliency\']);system(s);
s=strcat('mkdir ',['.\result\' Mode '\MotionSaliency\']);system(s);
s=strcat('mkdir ',['.\result\' Mode '\BeforeLowRank\']);system(s);
s=strcat('mkdir ',['.\result\' Mode '\AfterLowRank\']);system(s);
s=strcat('mkdir ',['.\result\' Mode '\FinalSaliency\']);system(s);
s=strcat('mkdir ',['.\result\' Mode '\Original\']);system(s);
Files = dir(fullfile(path,['*.' Type]));
LengthFiles = length(Files);
ImgIndex = 1;
%BatchSize = 8;
BatchInitSpan = 9;
BatchNum = floor((LengthFiles-3)/(BatchInitSpan-1));
BatchResidual = mod(LengthFiles-3,BatchInitSpan-1);
BatchSize =computeBatchSize(LengthFiles,BatchInitSpan);
index = 1;
% while(1)
%     if(BatchResidual==0)
%         break;
%     end
%     BatchSize{index} = BatchSize{index} + 1;
%     BatchResidual = BatchResidual-1;
%     index = index + 1;
%     if(index>BatchNum)
%         index = 1;
%     end
% end
BatchSubIndex = 1;
sigma_s = 100;sigma_r = 0.05;%Smoothing Parameter
BatchIndex = 0;
binNum = 5;
bin = 1/binNum;
CompressedAppearanceModel = zeros(1000,4);%[x y z Saliency]
CompressedAppearanceModelSize = 0;
AppearanceModelIndex = 1;
AppearanceModelMaxSize = 600;
EndFlag = 0;

%%
%Step 2: Begin Load Images
while(ImgIndex<=LengthFiles)
   BatchIndex = BatchIndex + 1;
   fprintf('compute the SP structure of images: '); 
   fprintf('%d ', BatchIndex); 
   ImageName = Files(ImgIndex).name;
   %ImageName = [num2str(ImgIndex) '.jpg'];
   ImagePath = [path ImageName];
   ImageNameContainer{BatchIndex} = ImageName;
   temp = imread(ImagePath);
   oW=size(temp,1);oH=size(temp,2);%Preserve ImageSize
   temp = imresize(temp,[300 300]);%Down Sampling
   I{BatchIndex} = im2double(temp);
   ISmoothed{BatchIndex} = RF(im2double(temp), sigma_s, sigma_r);%Image Preprocessing/Smoothing
   [SP{BatchIndex}] = getSPInfo(ISmoothed{BatchIndex},(ISmoothed{BatchIndex}));%get Superpixel Structure Information
   %imshow(SPMap,[]);
   %imwrite(frame2im(getframe(gcf)),['.\result\' num2str(ImgIndex),'.jpg']);
   fprintf('successed\n'); 
   [W H] = size(I{1});
   
%%
%Step 3: Begin Saliency Detection
   if(BatchIndex==BatchSize{BatchSubIndex}+1)
       N = BatchSize{BatchSubIndex};
       BatchSubIndex = BatchSubIndex+1;
       BatchIndex = 0;
       %Feature Extraction
       [color,motion,location,d,vx,vy] = getHistogramFeatures(I,SP);

       %N = size(SP,2)-1; %　total image number, column number of W

       %if(exist('LastBatchSaliencyRecord'))
       %    N = N + 1;
       %end
       K = -inf;
       for i=1:N
          K = max(K,SP{i}.SuperPixelNumber);% Identical Super Pixel Number
       end
       %K = max(K,lastK);

       MotionDim = size(motion{1}{1},2);
       ColorDim = size(color{1}{1},2);
       LocationDim = size(location{1}{1},2);
       clear BoundaryMask;
       %Load Input Data
       for i=1:N%Initial Containers
           templm = zeros(1,MotionDim+LocationDim);
           tempcm = zeros(1,MotionDim+ColorDim);
           tempc = zeros(1,ColorDim);
           for j=1:SP{i}.SuperPixelNumber%Normal Nodes
               DcInit{i}{j} = color{i}{j};
               BoundaryMask{i}{j} = shiftdim(SP{i}.BoundaryIndex(1,j));
           end
           for j=SP{i}.SuperPixelNumber+1:K%Dummy Nodes
               DcInit{i}{j} = tempc;
               color{i}{j} = [1 1 1];
               motion{i}{j} = [1 1];
               location{i}{j} = [300 300];
               BoundaryMask{i}{j} = [0];
           end
       end
       
       MotionGradientMatrix = getMotionGradient(vx,vy,ISmoothed,20);%get Contour Information(MotionGradient.*ColorGradient)
       
       BoundaryMaskMatrix = [];
       for i=1:N
           BoundaryMaskMatrix = [BoundaryMaskMatrix cell2mat(BoundaryMask{i})'];%Supress Boundary Superpixels
       end
       
       %Estimate the smooth range
       MotionGradientMatrix2 = getMotionGradient(vx,vy,ISmoothed,100);%get Contour Information(MotionGradient.*ColorGradient)
       SmoothRange = 0;
       for i=1:N
           CenterLocation = mean(MotionGradientMatrix2{i},2);
           SmoothRange = SmoothRange + mean(sum(abs(bsxfun(@minus,MotionGradientMatrix2{i},CenterLocation))));
       end
       SmoothRange = SmoothRange/N;
       if(exist('lastSmoothRange'))
          SmoothRange = 0.8*lastSmoothRange + 0.2*SmoothRange;
       end
       SmoothRange = max(SmoothRange,20);SmoothRange = min(SmoothRange,50);
       lastSmoothRange = SmoothRange;
       %SmoothRange = 30;

%%
%Compute the Motion Contrast and Color Contrast(Initial Saliency Clues)
       ColorSaliency = zeros(K,N);
       MotionSaliencyMatrix = zeros(K,N);
       for i=1:N
           for l=1:SP{i}.SuperPixelNumber
               lMotion =  motion{i}{l};
               lColor =  color{i}{l};
               lLocation = [SP{i}.MiddlePoint(l,1:2)];
               ValidNum1 = 0; ValidNum2 = 0; ValidNum3 = 0; ValidNum4 = 0;
               MotionScale1 = 0; 
               ColorScale1 = 0;
               T1 = 0; T2 = 0;
               weight = 0;
               for r=1:SP{i}.SuperPixelNumber
                   rMotion =  motion{i}{r};
                   rColor =  color{i}{r};
                   rLocation = [SP{i}.MiddlePoint(r,1:2)];
                   LocationDist = (sum(abs(rLocation-lLocation)));
                   MotionDist = (sum(abs(lMotion-rMotion)));
                   ColorDist = sqrt(sum((lColor-rColor).^2));
                   ring = min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},rLocation')),1));
                   if(LocationDist>ring && LocationDist<300)
                       MotionScale1 = MotionScale1 + MotionDist/(LocationDist+1);
                       ColorScale1 = ColorScale1 + ColorDist/(LocationDist+1);
                       %MotionScale1 = MotionScale1 + MotionDist*exp(-LocationDist/ring);
                       %ColorScale1 = ColorScale1 + ColorDist*exp(-LocationDist/ring);
                       %weight =  weight + exp(-LocationDist/ring);
                   end
                   if(LocationDist<300)
                       %ColorScale1 = ColorScale1 + ColorDist/(LocationDist+1);
                   end
               end
               MotionSaliencyMatrix(l,i) = MotionScale1;
               ColorSaliency(l,i) = ColorScale1;
           end
       end
       MotionSaliencyMatrix = MotionSaliencyMatrix.*BoundaryMaskMatrix;
       oM = MotionSaliencyMatrix.*ColorSaliency;
      
       TempColorSaliency = ColorSaliency;
       TempMotionSaliencyMatrix = MotionSaliencyMatrix;
       %Normalize ColorSaliency
       for i=1:N
           maxValue = max(TempColorSaliency(:,i));
           minValue = min(TempColorSaliency(:,i));
           TempColorSaliency(:,i) = (TempColorSaliency(:,i)-minValue)./(maxValue-minValue);
       end
      
       for i=1:N
           maxValue = max(TempMotionSaliencyMatrix(:,i));
           minValue = min(TempMotionSaliencyMatrix(:,i));
           TempMotionSaliencyMatrix(:,i) = (TempMotionSaliencyMatrix(:,i)-minValue)./(maxValue-minValue);
       end
      
       %Refine ColorSaliency
       for i=1:N
           for j=1:K
           	  if(TempColorSaliency(j,i) - TempMotionSaliencyMatrix(j,i)>0.5)
                  ColorSaliency(j,i) = ColorSaliency(j,i)*0.5;
              end
              if(TempMotionSaliencyMatrix(j,i) - TempColorSaliency(j,i)>0.5)
                  ColorSaliency(j,i) = mean(ColorSaliency(:,i))*2; 
              end
           end
       end
       if(exist('AppearanceModel'))
           for i=1:N
               for j=1:K
                   ColorValue = color{i}{j};
                   AContrast = 0; BContrast = 0;
                   minA = inf; minB = inf;
                   AppearanceModelLength = size(AppearanceModel,2);
                   for jj=1:AppearanceModelLength
                      dist = sqrt(sum((ColorValue-AppearanceModel{jj}).^2)); 
                      AContrast = AContrast + dist;
                      if(minA>dist)
                          minA = dist;
                      end
                   end
                   for jj=1:BackgroundModelLength
                      dist = sqrt(sum((ColorValue-BackgroundModel{jj}).^2));
                      BContrast = BContrast + dist;
                      if(minB>dist)
                          minB = dist;
                      end
                   end
                   %CurrentSaliencyDegree = BContrast*AppearanceModelLength/(AContrast*BackgroundModelLength);
                   %if(AContrast/AppearanceModelLength>BContrast/BackgroundModelLength)%belong to foreground
                   %if(MotionSaliencyMatrix(j,i)<0.5)
                   if(minA>minB*1.1 || AContrast/AppearanceModelLength>BContrast/BackgroundModelLength)%belong to foreground
                       ColorSaliency(j,i) = ColorSaliency(j,i) * 0.5;
                   end
                   %end
               end
           end
       end
%}
%%
%Smooth Intial Saliency Clues
       MotionContrastSmoothed = zeros(K,N);
       ColorContrastSmoothed = zeros(K,N);
       for i=1:N
           [W H] = size(I{1});
           ColorResult = zeros(W,H/3);
           MotionResult = zeros(W,H/3);
           FusionedResult = zeros(W,H/3);
           for l=1:SP{i}.SuperPixelNumber
               lColor = [SP{i}.MiddlePoint(l,3:5)];
               lLocation = [SP{i}.MiddlePoint(l,1:2)];
               TotalWeight = 0;
               MotionSaliencySum = 0;
               ColorSaliencySum = 0;
               ValidNum = 0;
               for r=1:SP{i}.SuperPixelNumber
                   rColor =  [SP{i}.MiddlePoint(r,3:5)];
                   rLocation = [SP{i}.MiddlePoint(r,1:2)];
                   ColorDist = sqrt(sum((rColor-lColor).^2));
                   LocationDist = (sum(abs(rLocation-lLocation)));
                   if(LocationDist<max(min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},rLocation')),1)),SmoothRange))%Shortest Dist to the MotionGradientMatrix
                       ValidNum = ValidNum+1;
                       weight = exp(-ColorDist*15);
                       TotalWeight = TotalWeight + weight;
                       MotionSaliencySum = MotionSaliencySum + MotionSaliencyMatrix(r,i)*weight; 
                       ColorSaliencySum = ColorSaliencySum + ColorSaliency(r,i)*weight; 
                         if(i+1<=N && r<=SP{i+1}.SuperPixelNumber)
                            rColor2 =  [SP{i+1}.MiddlePoint(r,3:5)];
                            ColorDist2 = sqrt(sum((rColor2-lColor).^2));
                            weight2 = exp(-ColorDist2*15);
                            TotalWeight = TotalWeight + weight2;
                            MotionSaliencySum = MotionSaliencySum + MotionSaliencyMatrix(r,i+1)*weight2; 
                            ColorSaliencySum = ColorSaliencySum + ColorSaliency(r,i+1)*weight2;
                         end
                         if(i-1>=1 && r<=SP{i-1}.SuperPixelNumber)
                            rColor3 =  [SP{i-1}.MiddlePoint(r,3:5)];
                            ColorDist3 = sqrt(sum((rColor3-lColor).^2));
                            weight3 = exp(-ColorDist3*15);
                            TotalWeight = TotalWeight + weight3;
                            MotionSaliencySum = MotionSaliencySum + MotionSaliencyMatrix(r,i-1)*weight3;  
                            ColorSaliencySum = ColorSaliencySum + ColorSaliency(r,i-1)*weight3;
                         end 
                       %}
                   end 
               end
               ClusteringPixelNumber = SP{i}.ClusteringPixelNum(1,l);
               MotionSaliencySum = MotionSaliencySum/(TotalWeight);
               ColorSaliencySum = ColorSaliencySum/(TotalWeight);
               MotionContrastSmoothed(l,i) = MotionSaliencySum;
               ColorContrastSmoothed(l,i) = ColorSaliencySum;
            
               for j=1:ClusteringPixelNumber%for Test Only
                  XIndex= SP{i}.Clustering(l,j,1);
                  YIndex= SP{i}.Clustering(l,j,2);
                  %Result(XIndex,YIndex) = MotionSaliencySum.*ColorSaliencySum;
                  ColorResult(XIndex,YIndex) = ColorSaliencySum;
                  MotionResult(XIndex,YIndex) = MotionSaliencySum;
                  %ColorResult(XIndex,YIndex) = ColorSaliency(l,i);
                  %MotionResult(XIndex,YIndex) = MotionSaliencyMatrix(l,i);
                  FusionedResult(XIndex,YIndex) = MotionSaliencySum.*ColorSaliencySum;
                  %Result(XIndex,YIndex) = Sint(l,i);
                  %Result(XIndex,YIndex) = MotionSaliencySum*ColorSaliencySum*MotionSaliencySum;
               end
           end
           index = find(ColorResult==0);
           ColorResult = abs(ColorResult);
           ColorResult(index) = mean(mean(ColorResult));
           ColorResult = MatrixNormalization(ColorResult);
           MotionResult = MatrixNormalization(MotionResult);
           FusionedResult = MatrixNormalization(FusionedResult);
           ColorResult = imresize(ColorResult,[oW oH]);
           MotionResult = imresize(MotionResult,[oW oH]);
           FusionedResult = imresize(FusionedResult,[oW oH]);
           imwrite(FusionedResult*1.5,['.\result\' Mode '\BeforeLowRank\' ImageNameContainer{i}]);
           imwrite(ColorResult*1.5,['.\result\' Mode '\ColorSaliency\' ImageNameContainer{i}]);
           imwrite(MotionResult*1.5,['.\result\' Mode '\MotionSaliency\' ImageNameContainer{i}]);
       end
       MotionSaliencyMatrix =  MotionContrastSmoothed.*ColorContrastSmoothed; 
       MotionSaliencyMatrix = MotionSaliencyMatrix.*BoundaryMaskMatrix;
       %MotionSaliencyMatrix =  (MotionContrastSmoothed + ColorContrastSmoothed)./2; 
       %MotionSaliencyMatrix =  MotionContrastSmoothed;
       [IndexX IndexY] = find(isnan(MotionSaliencyMatrix)==1);%Eliminate Ill SLIC SuperPixels
       MotionSaliencyMatrix(IndexX,IndexY) = 0;
       
%%
%Locate the Coarse Motion Areas
       maxValue = max(MotionSaliencyMatrix);
       minValue = min(MotionSaliencyMatrix);
       for i=1:N
          MotionSaliencyMatrix(:,i) = (MotionSaliencyMatrix(:,i)-minValue(1,i))./(maxValue(1,i)-minValue(1,i));
       end
       if(BatchSubIndex==2)
           rate = 2.0;
       else
           rate = 2.0;
       end
       temp = MotionSaliencyMatrix;
       [index value] = find(MotionSaliencyMatrix(:)<0.05);
       temp(index) = 0;
       temp = sum(abs(temp),2);
       Threshold = mean(sum(abs(MotionSaliencyMatrix),2));
       if(exist('LastBatchSaliencyRecord'))
           %temp2 = LastBatchSaliencyRecord(:,end);
           %maxValue = max(temp2);minValue = min(temp2);
           %temp2 = (temp2-minValue)/(maxValue-minValue);
           %temp = (N-1)/N*temp + 0.5*temp2;
       end
       count = 0;
       while(count<50)
           count = count + 1;
           [index value] = find(temp>Threshold*rate);
           MotionMask = zeros(K,1);
           MotionMask(index,1) = 1;
           [index value] = find(MotionMask==1);
           MotionMaskLength = size(index,1);
           if(~exist('lastMotionMaskLength'))
               lastMotionMaskLength = MotionMaskLength;
           end
           if(abs(MotionMaskLength-lastMotionMaskLength)<=15)%Minimal Number of Selected Superpixels
               break;
           end
           if(MotionMaskLength<lastMotionMaskLength)
               rate = rate * 0.95; 
           else
               rate = rate * 1.05;
           end
       end
       %}
    
%%
%Inital the Da(the Gradient-aware Constraint)
       %minValuePool = zeros(1,N);
       %maxValuePool = zeros(1,N);
       minValue = inf;maxValue = -inf;
       for i=1:N
           for j=1:SP{i}.SuperPixelNumber
               temp = 0;
               cLocation = SP{i}.MiddlePoint(j,1:2);
               DaInit{i}{j} = min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},cLocation')),1));
               if(DaInit{i}{j}>maxValue)
                   maxValue = DaInit{i}{j};
               end
               if(DaInit{i}{j}<minValue)
                   minValue = DaInit{i}{j};
               end
           end
           %minValuePool(1,i) = minValue;
           %maxValuePool(1,i) = maxValue;
       end
       
       %Normalization
       for i=1:N
           %minValue = minValuePool(1,i);
           %maxValue = maxValuePool(1,i);
           for j=1:SP{i}.SuperPixelNumber
               if(maxValue-minValue~=0)
                   DaInit{i}{j} = (DaInit{i}{j}-minValue)/(maxValue-minValue);
               else
                   DaInit{i}{j} = 0;
               end
           end
       end
       
%%
      for i=1:N
          for j=1:SP{i}.SuperPixelNumber
              DlInit{i}{j} = location{i}{j};
          end
      end
%%
%Initial the Dp(the Saliency Clue Constraint)
      for i=1:N
          for j=1:SP{i}.SuperPixelNumber
              DpInit{i}{j} = [MotionSaliencyMatrix(j,i)];
          end
      end

%%
%get Sub Features
      CostPenalty = zeros(MotionMaskLength,N);%for Saliency Adjustment
      ro = 0.05;
      for i=1:N
          P{i} = [1:1:MotionMaskLength];%Purmutation Matrix
      end
      %Formulate Sub Feature Matrix
      Dc = ones(MotionMaskLength*ColorDim,N); 
      Dp = zeros(MotionMaskLength,N);
      Da = zeros(MotionMaskLength,N); 
      Dl = zeros(MotionMaskLength*2,N);
      
      OKContainer = ones(MotionMaskLength,N);
      OKIndex = double(repmat([1:1:K]',1,N));
      for i=1:N%Fill
          count = 0;
          clear DcBox;
          clear OKBox;
          clear DpBox;
          clear DaBox;
          clear DlBox;
          for j=1:SP{i}.SuperPixelNumber
             index = OKIndex(j,i);
             if(MotionMask(j,1)==1)
                count = count + 1;
                DcBox{count} = DcInit{i}{index};
                DpBox{count} = DpInit{i}{index};
                DaBox{count} = DaInit{i}{index};
                DlBox{count} = DlInit{i}{index};
                OKBox{count} = index;
             end
          end
          Dc(1:count*ColorDim,i) = cell2mat(DcBox)';
          Dp(1:count,i) = cell2mat(DpBox)';
          Da(1:count,i) = cell2mat(DaBox)';
          Dl(1:count*2,i) = cell2mat(DlBox)';
          OKContainer(1:count,i) = cell2mat(OKBox)';
      end    
      OKIndex = OKContainer;
      InitDc = Dc;%Save Initial Sub Matrix
      InitDp = Dp;
      InitDa = Da;
      InitDl = Dl;
      InitOKIndex = OKIndex;
      
%%
%Construct Structure Graph
      NeighborGraphBox = zeros(MotionMaskLength,MotionMaskLength);
      NeighborColorDist = zeros(MotionMaskLength,MotionMaskLength);
      for i=1:N
          for l=1:MotionMaskLength
              lLocation = Dl(l,:);
              lColor =  [SP{i}.MiddlePoint(OKIndex(l,i)',3:5)];
              for r=1:MotionMaskLength
                  rColor = [SP{i}.MiddlePoint(OKIndex(r,i)',3:5)];
                  ColorDist = sqrt(sum((rColor-lColor).^2));
                      rLocation = Dl(r,:);
                      Dist = sum(abs(lLocation-rLocation));
                      if(Dist>0.2)
                          NeighborGraphBox(l,r) = 0;
                      else
                          %NeighborGraphBox(l,r) = exp(-Dist*0.5);
                         if(l~=r)
                          NeighborGraphBox(l,r) = 1;
                         end
                      end
              end
          end
          NeighborGraph{i} = NeighborGraphBox;
      end

%%
%Initial ADMM Parameters
      Y8 = zeros(MotionMaskLength*ColorDim,N);
      Y10 = zeros(MotionMaskLength,N);
      Y11 = zeros(MotionMaskLength,N);
      Y12 = zeros(MotionMaskLength*2,N);
    
      Lc = zeros(MotionMaskLength*ColorDim,N); Ec = Lc;
      Lp = zeros(MotionMaskLength,N); Ep = Lp;
      La = zeros(MotionMaskLength,N); Ea = La;
      Ll = zeros(MotionMaskLength*2,N); El = Ll;
    
      %solve optimization
      iter = 0;
      fprintf('enter ADMM\n');
      while(1)
          iter = iter + 1;
          temp = Dc-Ec+Y8/ro;
          [U S V] = svd(temp, 'econ');
          Lc = U*diag(pos(diag(S)-0.1/ro))*V';
            
          %update Lp
          temp = Dp-Ep+Y10/ro;
          [U S V] = svd(temp, 'econ');
          Lp = U*diag(pos(diag(S)-0.05/ro))*V';
            
          %update La
          temp = Da-Ea+Y11/ro;
          [U S V] = svd(temp, 'econ');
          La = U*diag(pos(diag(S)-0.05/ro))*V';
          
          %Ll
          temp = Dl;
          for i=2:N-1
             for j=1:MotionMaskLength*2
                 temp(j,i) = (Dl(j,i-1)+Dl(j,i+1))/2;
             end
          end
          Ll = temp;
           
          %update Ec
          temp = Dc-Lc+Y8/ro;
          Ec = sign(temp) .* pos( abs(temp) -0.03/ro);
            
          %update Ep
          temp = Dp-Lp+Y10/ro;
          Ep = sign(temp) .* pos( abs(temp) -0.01/ro);
            
          %update Ea
          temp = Da-La+Y11/ro;
          Ea = sign(temp) .* pos( abs(temp) -0.01/ro);
          
          %update El
          temp = Dl-Ll+Y12/ro;
          El = sign(temp) .* pos( abs(temp) -500/ro);
          
          DcObj = (Lc + Ec - Y8/ro);
          DpObj = (Lp + Ep - Y10/ro);
          DaObj = (La + Ea - Y11/ro); 
          DlObj = (Ll + El - Y12/ro); 
          for i=1:N
              DcCostMatrix = zeros(MotionMaskLength,MotionMaskLength);
              DpCostMatrix = zeros(MotionMaskLength,MotionMaskLength);
              DaCostMatrix = zeros(MotionMaskLength,MotionMaskLength);
              DlCostMatrix = zeros(MotionMaskLength,MotionMaskLength);
              for l = 1:MotionMaskLength
                  DcTempCell = DcObj((l-1)*ColorDim+1:l*ColorDim,i);
                  DpTempCell = DpObj((l-1)+1:l,i);
                  DaTempCell = DaObj((l-1)+1:l,i);
                  DlTempCell = DlObj((l-1)*2+1:l*2,i);
                  for r = 1:MotionMaskLength % euclidean distance
                      DcCostMatrix(l,r) = (sum(abs(InitDc((r-1)*ColorDim+1:r*ColorDim,i)-DcTempCell)));
                      DpCostMatrix(l,r) = (sum(abs(InitDp((r-1)+1:r,i)-DpTempCell)));
                      DaCostMatrix(l,r) = (sum(abs(InitDa((r-1)+1:r,i)-DaTempCell)));
                      DlCostMatrix(l,r) = (sum(abs(InitDl((r-1)*2+1:r*2,i)-DlTempCell)));
                  end
              end
              FinalCostMatrix = 1*DcCostMatrix + 1*DpCostMatrix;
              FinalCostMatrixContainer{i} = FinalCostMatrix;
          end
           
           %Introduce Structure Info
           TempContainer = zeros(MotionMaskLength,MotionMaskLength);
           for i=1:N-1
              for j=1:MotionMaskLength
                  temp = FinalCostMatrixContainer{i+1}.*repmat(NeighborGraph{i}(j,:),[MotionMaskLength 1]);
                  [index] = find(temp==0);
                  temp(index) = 0;
                  ColorMask = zeros(1,MotionMaskLength);
                  lColor = [SP{i}.MiddlePoint(OKIndex(j,i)',3:5)];
                  [index] = find(NeighborGraph{i}(j,:)~=0);%neightbor nodes
                  for jj=1:size(index,2)
                      rcolor = [SP{i+1}.MiddlePoint(OKIndex(index(1,jj),i+1)',3:5)];
                      ColorDist = sqrt(sum((rColor-lColor).^2));
                      weight = exp(-ColorDist*15);
                      ColorMask(1,jj) = weight;
                  end
                  temp = temp.*repmat(ColorMask,[MotionMaskLength 1]);
                  if(sum(ColorMask)~=0)
                      TempContainer(j,:) = sum(temp,2)./sum(ColorMask);
                  else
                      TempContainer(j,:) = sum(temp,2);
                  end
                      
                  if(size(index,2)==0)
                      TempContainer(j,:) = TempContainer(j,:) + 999;
                  end    
              end
              WeightedCostMatrix1{i} = TempContainer;
           end
           WeightedCostMatrix1{N} = FinalCostMatrixContainer{N};
           
           TempContainer = zeros(MotionMaskLength,MotionMaskLength);
           for i=2:N
              for j=1:MotionMaskLength
                  temp = FinalCostMatrixContainer{N+1-i}.*repmat(NeighborGraph{N+2-i}(j,:),[MotionMaskLength 1]);
                  [index] = find(temp==0);
                  temp(index) = 0;
                  ColorMask = zeros(1,MotionMaskLength);
                  lColor = [SP{N+2-i}.MiddlePoint(OKIndex(j,N+2-i)',3:5)];
                  [index] = find(NeighborGraph{N+2-i}(j,:)~=0);%neightbor nodes
                  for jj=1:size(index,2)
                      rcolor = [SP{N+1-i}.MiddlePoint(OKIndex(index(1,jj),N+1-i)',3:5)];
                      ColorDist = sqrt(sum((rColor-lColor).^2));
                      weight = exp(-ColorDist*15);
                      ColorMask(1,jj) = weight;
                  end
                  temp = temp.*repmat(ColorMask,[MotionMaskLength 1]);
                  if(sum(ColorMask)~=0)
                      TempContainer(j,:) = sum(temp,2)./sum(ColorMask);
                  else
                      TempContainer(j,:) = sum(temp,2);
                  end
                  if(size(index,2)==0)
                      TempContainer(j,:) = TempContainer(j,:) + 999;
                  end  
                  %TempContainer(j,:) = min(temp');
                  %TempContainer(j,:) = mean(temp');
              end
              WeightedCostMatrix2{N+2-i} = TempContainer;
           end
           WeightedCostMatrix2{1} = FinalCostMatrixContainer{1};
           
           % Hungarian algorithm,0.002s
           for i=1:N
              WeightedCostMatrix{i} = 1.0*(WeightedCostMatrix1{i}+WeightedCostMatrix2{i}) + FinalCostMatrixContainer{i};
              [index,cost] = munkres(WeightedCostMatrix{i}); 
              %[index,cost] = munkres(FinalCostMatrixContainer{i}); 
              P{i} = index; %update
           end
      
           %Update
           for i=1:N%Select
              OKIndex(:,i) = InitOKIndex(P{i},i);
           end
           clear DcBox;
           clear DpBox;
           for i=1:N
               for j=1:MotionMaskLength
                   index = OKIndex(j,i);
                   DcBox{j} = DcInit{i}{index};
                   DpBox{j} = DpInit{i}{index};
                   DaBox{j} = DaInit{i}{index};
                   DlBox{j} = DlInit{i}{index};
               end
               Dc(:,i) = cell2mat(DcBox)';
               Dp(:,i) = cell2mat(DpBox)';
               Da(:,i) = cell2mat(DaBox)';
               Dl(:,i) = cell2mat(DlBox)';
           end

           Y8 = Y8 + ro*(Dc-Lc-Ec);
           Y10 = Y10 + ro*(Dp-Lp-Ep);
           Y11 = Y11 + ro*(Da-La-Ea);
           Y12 = Y12 + ro*(Dl-Ll-El);
          
           %update ro
           ro = ro*1.05;
           
           if(iter>=200)
                break;
           end
           fprintf('Iteration %d\n',iter);
      end

%%
%Compute the CostPenalty
      Ec = abs(Ec);
      lastCostPenalty = CostPenalty-1;
      CostPenalty = zeros(MotionMaskLength,N);
      for i=1:N
          for j=1:MotionMaskLength
              temp = sum(Ec((j-1)*ColorDim+1:j*ColorDim,i)); 
              if(temp>0)
                  CostPenalty(find(P{i}==j),i) = 1;
              end
          end
      end
      
%%
%Saliency Adjustment
     Ds = zeros(MotionMaskLength,N);
     Saliency = MotionSaliencyMatrix;
     clear DsBox;
     [index1 value]= find(MotionMask==1);
     for i=1:N%Update Ds
         for j=1:MotionMaskLength
             index = OKIndex(j,i);
             DsBox{j} = MotionSaliencyMatrix(index,i);%Original Saliency Clue(MotionSaliency & ColorSaliency)
         end
         Ds(:,i) = cell2mat(DsBox)';
     end
     MeanDs = mean(Ds,2);
     temp = bsxfun(@minus,Ds,MeanDs);%CostPenalty(Superpixels with high Ec)
     BalanceMatrix = zeros(MotionMaskLength,N);
     for i=1:N
         for j=1:MotionMaskLength
             if(temp(j,i)>0)
                 BalanceMatrix(j,i) = MeanDs(j,1)*0.5;%Stay
             end
             if(temp(j,i)<0)
                 BalanceMatrix(j,i) = MeanDs(j,1)*2;%Compensate
             end
         end
     end
     Ds = (Ds-Ds.*CostPenalty)+BalanceMatrix.*CostPenalty;
     
     maxValue = max(max(Ds)); minValue = min(min(Ds));
     Ds = (Ds-minValue)./(maxValue-minValue);
     
     if(exist('LastBatchSaliencyRecord'))
        clear SaliencyPriorBox;
        SaliencyPrior = LastBatchSaliencyRecord(:,end);
        maxValue = max(SaliencyPrior); minValue = min(SaliencyPrior);
        SaliencyPrior = (SaliencyPrior-minValue)/(maxValue-minValue);
         for j=1:MotionMaskLength
             index = OKIndex(j,1);
             SaliencyPriorBox{j} = SaliencyPrior(index,1);%Original Saliency Clue(MotionSaliency & ColorSaliency)
         end
     end
     
     CurrentDs = zeros(MotionMaskLength,N);
     for i=1:N
         for j=1:MotionMaskLength
             temp = Ds(j,i);
             SPIndex = OKIndex(j,i)';
             lColor = [SP{i}.MiddlePoint(SPIndex,3:5)];
             pColor = [SP{N}.MiddlePoint(OKIndex(j,N)',3:5)];
             lLocation = [SP{i}.MiddlePoint(SPIndex,1:2)];
             pLocation = [SP{N}.MiddlePoint(OKIndex(j,N)',1:2)];
             TotalWeight = 0;
             SaliencySum = 0;
             for ii=1:N
                 SPIndex = OKIndex(j,ii)';
                 rColor = [SP{ii}.MiddlePoint(SPIndex,3:5)];
                 rLocation = [SP{ii}.MiddlePoint(SPIndex,1:2)];
                 ColorDist = (sum(abs(rColor-lColor)));
                 weight = exp(-ColorDist*5);
                 TotalWeight = TotalWeight + weight;
                 SaliencySum = SaliencySum + weight*Ds(j,ii);
             end
             if(exist('LastBatchSaliencyRecord'))
                 ColorDist = (sum(abs(pColor-lColor)));
                 weight = exp(-ColorDist*25);
                 LocationDist = sqrt(sum(abs(pLocation-lLocation).^2));
                 if(LocationDist<0.2)
                     TotalWeight = TotalWeight + weight;
                     SaliencySum = SaliencySum + 0.5*weight*SaliencyPriorBox{l};
                 end
                 %}
             end
             CurrentDs(j,i) = SaliencySum/TotalWeight;
         end
     end
     SS=sum(CurrentDs,2)';
     %SS=sum(Ds,2)';
     
     maxValue = max(SS); minValue = min(SS);
     SS = (SS-minValue)/(maxValue-minValue);
     
     if(exist('LastBatchSaliencyRecord'))
         SS = 0.7*SS + 0.3*cell2mat(SaliencyPriorBox);
         %SS = max(SS,cell2mat(SaliencyPriorBox));
     end
     
%%
%Compact the Saliency Distribution
     SS = SS - min(SS);
     
     [temp index] = sort(SS,'descend');
     [temp2 index2] = sort(SS-mean(SS)*2,'descend');
     [value index2] = find(temp2>0);
     SmoothScale = max(size(index2,2),3);
     lowBound = temp(1,SmoothScale);
     upBound = temp(1,1);
     SmoothDegree = (upBound - lowBound)/(SmoothScale-1);
     for i=1:SmoothScale-1
         temp(1,SmoothScale-i) = lowBound + SmoothDegree*i*0.5;
     end
     %SS(1,index(1:SmoothScale)) = temp(1:SmoothScale);
   
     for FrameIndex = 1:N
         for i=1:K
             SaliencyAssign{FrameIndex}{i} = 0;
             SaliencyAssign2{FrameIndex}{i} = 0;
         end
     end
     for FrameIndex = 1:N
         [W H] = size(I{FrameIndex});
         Result = zeros(W,H/3);
         SuperPixelNumber = SP{FrameIndex}.SuperPixelNumber;
         for i=1:MotionMaskLength
             SPIndex = OKIndex(i,FrameIndex)';
             SaliencyValue = SS(1,i);
             ClusteringPixelNumber = SP{FrameIndex}.ClusteringPixelNum(1,SPIndex);
             SaliencyAssign{FrameIndex}{SPIndex} = SaliencyValue;
             for j=1:ClusteringPixelNumber
                 XIndex= SP{FrameIndex}.Clustering(SPIndex,j,1);
                 YIndex= SP{FrameIndex}.Clustering(SPIndex,j,2);
                 Result(XIndex,YIndex) = max(SaliencyValue,Result(XIndex,YIndex));
             end
         end
         Result = MatrixNormalization(Result);
         Result = imresize(Result,[oW oH]);
         imwrite(Result*1.5,['.\result\' Mode '\Original\'  ImageNameContainer{FrameIndex}]);
     end
  
     
%%
%1st round smoothing
     fprintf('begin superpixel level smoothing...\n');
     for i=1:N
         [W H] = size(ISmoothed{FrameIndex});
         Result = zeros(W,H/3);
         for l=1:SP{i}.SuperPixelNumber
             lColor = [SP{i}.MiddlePoint(l,3:5)];
             lLocation = [SP{i}.MiddlePoint(l,1:2)];
             TotalWeight = 0;
             SaliencySum = 0;
             ValidNum = 0;
             for r=1:SP{i}.SuperPixelNumber
                 rColor =  [SP{i}.MiddlePoint(r,3:5)];
                 rLocation = [SP{i}.MiddlePoint(r,1:2)];
                 ColorDist = sqrt(sum((rColor-lColor).^2));
                 LocationDist = (sum(abs(rLocation-lLocation)));
                 if(LocationDist<max((min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},rLocation')),1))),SmoothRange))
                     ValidNum = ValidNum+1;
                     weight = exp(-ColorDist*15);
                     TotalWeight = TotalWeight + weight;
                     SaliencySum = SaliencySum + SaliencyAssign{i}{r}*weight;
                    
                     if(i+1<=N && r<=SP{i+1}.SuperPixelNumber)
                        rColor2 =  [SP{i+1}.MiddlePoint(r,3:5)];
                        ColorDist2 = sqrt(sum((rColor2-lColor).^2));
                        weight2 = exp(-ColorDist2*15);
                        TotalWeight = TotalWeight + weight2;
                        SaliencySum = SaliencySum + SaliencyAssign{i+1}{r}*weight2; 
                     end
                     if(i-1>=1 && r<=SP{i-1}.SuperPixelNumber)
                        rColor3 =  [SP{i-1}.MiddlePoint(r,3:5)];
                        ColorDist3 = sqrt(sum((rColor3-lColor).^2));
                        weight3 = exp(-ColorDist3*15);
                        TotalWeight = TotalWeight + weight3;
                        SaliencySum = SaliencySum + SaliencyAssign{i-1}{r}*weight3;  
                     end
                     %}
                 end 
             end
             SaliencyAssign5{i}{l}= SaliencySum/TotalWeight;
             ClusteringPixelNumber = SP{i}.ClusteringPixelNum(1,l);
             for j=1:ClusteringPixelNumber
             	XIndex= SP{i}.Clustering(l,j,1);
                YIndex= SP{i}.Clustering(l,j,2);
                Result(XIndex,YIndex) = SaliencySum/TotalWeight;
             end
         end
         Result = MatrixNormalization(Result);
         Result = imresize(Result,[oW oH]);
         imwrite(Result*1.5,['.\result\' Mode '\AfterLowRank\'  ImageNameContainer{i}]);
     end
         
          
%%
%Final Smoothing & Pixel-wise Smoothing (2nd round smoothing)
       fprintf('begin pixel level smoothing...\n');
       minK = inf;
       for i=1:N
          minK = min(minK,SP{i}.SuperPixelNumber);% Identical Super Pixel Number
       end
      LastBatchSaliencyRecord = zeros(K,N);
      for i=1:N
         [W H] = size(ISmoothed{FrameIndex});
         Result = zeros(W,H/3);
         for l=1:SP{i}.SuperPixelNumber
             lColor = [SP{i}.MiddlePoint(l,3:5)];
             lLocation = [SP{i}.MiddlePoint(l,1:2)];
             TotalWeight = 0;
             SaliencySum = 0;
             ValidNum = 0;
             binIndex = floor(pos((lColor-0.000001)./bin))+1;
             RIndex = binIndex(1,1);GIndex = binIndex(1,2);BIndex = binIndex(1,3);
             %ring = max(min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},lLocation')),1)),30);
             for r=1:SP{i}.SuperPixelNumber
                 rColor =  [SP{i}.MiddlePoint(r,3:5)];
                 rLocation = [SP{i}.MiddlePoint(r,1:2)];
                 ColorDist = sqrt(sum((rColor-lColor).^2));
                 LocationDist = (sum(abs(rLocation-lLocation)));
                 if(LocationDist<max((min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},rLocation')),1))),SmoothRange*0.5))
                     ValidNum = ValidNum+1;
                     weight = exp(-ColorDist*15);
                     TotalWeight = TotalWeight + weight;
                     SaliencySum = SaliencySum + SaliencyAssign5{i}{r}*weight;
                    
                     if(i+1<=N)
                         if(r<=SP{i+1}.SuperPixelNumber)
                            rColor2 =  [SP{i+1}.MiddlePoint(r,3:5)];
                            ColorDist2 = sqrt(sum((rColor2-lColor).^2));
                            weight2 = exp(-ColorDist2*15);
                            TotalWeight = TotalWeight + weight2;
                            SaliencySum = SaliencySum + SaliencyAssign5{i+1}{r}*weight2; 
                         end
                     end
                     if(i-1>=1)
                         if(r<=SP{i-1}.SuperPixelNumber)
                            rColor3 =  [SP{i-1}.MiddlePoint(r,3:5)];
                            ColorDist3 = sqrt(sum((rColor3-lColor).^2));
                            weight3 = exp(-ColorDist3*15);
                            TotalWeight = TotalWeight + weight3;
                            SaliencySum = SaliencySum + SaliencyAssign5{i-1}{r}*weight3;  
                         end
                     end
                     %}
                 end 
             end
             LastBatchSaliencyRecord(l,i) = SaliencySum/TotalWeight;
             ClusteringPixelNumber = SP{i}.ClusteringPixelNum(1,l);
             for j=1:ClusteringPixelNumber
                XIndex= SP{i}.Clustering(l,j,1);
                YIndex= SP{i}.Clustering(l,j,2);
                Result(XIndex,YIndex) = SaliencySum/TotalWeight;
             end
         end
        
         %Pixel-wise Assignment
         LocalSize = 20;
         ImageSize = 300;
         ResultBox1 = imresize(Result,[ImageSize,ImageSize]);
         ResultBox2 = imresize(I{i},[ImageSize,ImageSize]);
         ResultBox3 = zeros(ImageSize,ImageSize);
         ResultBox4 = zeros(ImageSize,ImageSize);
         ResultBox5 = zeros(ImageSize,ImageSize);
         for ii=1:ImageSize
             for jj=1:ImageSize
                 PixelValue = ResultBox2(ii,jj,:);

                 SaliencyPatch = ResultBox1(max(ii-LocalSize,1):min(ii+LocalSize,ImageSize),max(jj-LocalSize,1):min(jj+LocalSize,ImageSize));
                 RChannel = ResultBox2(max(ii-LocalSize,1):min(ii+LocalSize,ImageSize),max(jj-LocalSize,1):min(jj+LocalSize,ImageSize),1)-ResultBox2(ii,jj,1);
                 GChannel = ResultBox2(max(ii-LocalSize,1):min(ii+LocalSize,ImageSize),max(jj-LocalSize,1):min(jj+LocalSize,ImageSize),2)-ResultBox2(ii,jj,2);
                 BChannel = ResultBox2(max(ii-LocalSize,1):min(ii+LocalSize,ImageSize),max(jj-LocalSize,1):min(jj+LocalSize,ImageSize),3)-ResultBox2(ii,jj,3);      
                 ColorDistMatrix = abs(RChannel)+abs(GChannel)+abs(BChannel);
              
                 WeightMatirx1 = exp(-sum(abs(ColorDistMatrix),3)*25);
                 TotalWeight1 = sum(sum(WeightMatirx1));
                 ResultBox3(ii,jj) = sum(sum(WeightMatirx1.*SaliencyPatch))/TotalWeight1;
             end
         end
         ResultBox3 = MatrixNormalization(ResultBox3);
         ResultSave = ResultBox3;
         ResultBox3 = imresize(ResultBox3,[oW oH]);
         Result = MatrixNormalization(Result);
         imwrite(ResultBox3*2,['.\result\' Mode '\FinalSaliency\'  ImageNameContainer{i}]);
      end

%%
%Update AppearanceModel
     fprintf('begin update appearance model...\n');
     AppearanceModelNum = zeros(binNum,binNum,binNum);%用于记录选中次数
     FBThreshold = mean(mean(abs(LastBatchSaliencyRecord)));
     count1 = 0; count2 = 0;
     for i=1:N
         for j=1:SP{i}.SuperPixelNumber%traval all SP
             CurrentSaliencyDegree = LastBatchSaliencyRecord(j,i);
             ColorValue = color{i}{j};
             binIndex = floor(pos((ColorValue-0.000001)./bin))+1;
             RIndex = binIndex(1,1);GIndex = binIndex(1,2);BIndex = binIndex(1,3);
             if(CurrentSaliencyDegree>5.0*FBThreshold)%Foreground
                 count1 = count1+1;
                 Index = max(mod(AppearanceModelIndex,AppearanceModelMaxSize),1);
                 AppearanceModel{Index} = ColorValue;
                 AppearanceModelIndex = Index + 1;
             else%Background
                 count2 = count2+1;
                 BackgroundModel{count2} = ColorValue; 
             end
         end
     end
     %AppearanceModelLength = count1;
     BackgroundModelLength = count2;
     
%%
       rate = 1.5;
       temp = sum(abs(LastBatchSaliencyRecord),2);
       [index value] = find(temp>mean(temp)*rate);
       MotionMask = zeros(K,1);
       MotionMask(index,1) = 1;
       [index value] = find(MotionMask==1);
       MotionMaskLengthCurrent = size(index,1);
       if(exist('lastMotionMaskLength'))
           lastMotionMaskLength = floor(0.5*lastMotionMaskLength + 0.5*MotionMaskLengthCurrent);
       else
           lastMotionMaskLength = MotionMaskLengthCurrent;
       end
       fprintf('last MotionMaskLength: %d\n',lastMotionMaskLength);
     
       if(ImgIndex+4>LengthFiles)
           break;
       end

       ImgIndex = ImgIndex-1;
   end
  
   ImgIndex = ImgIndex+1;

   clc;
   close all;
end
