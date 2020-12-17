%参数d为histogram的维数

function [color,motion,location,d,vx,vy] = getHistogramFeatures(I,SP)


d = 12;

W = size(I{1},1);
H = size(I{1},2);

%计算各图像的motion
for i=1:size(I,2)-1
   [vx{i},vy{i}] = getMotionFeature(I{i},I{i+1});% motion feature 
   %MotionLength{i} = sqrt(vx{i}.^2+vy{i}.^2);%对vx，vy进行转换，变成长度和角度
   MotionLength{i} = abs(vx{i}+vy{i});
   MotionAngle{i} = atan(vy{i}./vx{i});
end

maxValue = max(max(max(cell2mat(vx))),max(max(cell2mat(vy))));% 归一化处理
minValue = min(min(min(cell2mat(vx))),min(min(cell2mat(vy))));
for i=1:size(I,2)-1
   if(maxValue-minValue~=0)
      vx{i} = (vx{i}-minValue)/(maxValue-minValue);
      vy{i} = (vy{i}-minValue)/(maxValue-minValue);
   end
end

MotionLengthMax = max(max(cell2mat(MotionLength)));% 归一化处理
MotionLengthMin = min(min(cell2mat(MotionLength)));
MotionAngleMax = max(max(cell2mat(MotionAngle)));% 归一化处理
MotionAngleMin = min(min(cell2mat(MotionAngle)));
for i=1:size(I,2)-1
   if(MotionLengthMax-MotionLengthMin~=0)
      MotionLength{i} = (MotionLength{i}-MotionLengthMin)/(MotionLengthMax-MotionLengthMin);
   end
   if(MotionAngleMax-MotionAngleMin~=0)
       MotionAngle{i} = (MotionAngle{i}-MotionAngleMin)/(MotionAngleMax-MotionAngleMin);
   end
end
%}

%计算各图像中，各像素点的颜色histogram
for i=1:size(I,2)-1
    %计算MotionContrast
    %MotionContrast = getMotionContrast(vx{i},vy{i});
    for j=1:SP{i}.SuperPixelNumber
        PixelPool = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),3:5));
        %color{i}{j} = getHistogram(PixelPool,d);% color histogram
        color{i}{j} = mean(abs(PixelPool));% color histogram
        %{
        SPIndex = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),1:2));
        MotionPool = zeros(size(SPIndex,1),2);
        for k=1:size(SPIndex,1)
           %MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2))];
           %计算大小(长度)与角度 
           %MotionPool(k,:) = [MotionLength{i}(SPIndex(k,1),SPIndex(k,2)) MotionAngle{i}(SPIndex(k,1),SPIndex(k,2))];
           MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2))];
        end
        motion{i}{j} = getHistogram(MotionPool,d);
        %motion{i}{j} = mean(MotionPool);
        %motion{i}{j} = mean(abs(MotionPool));
        %}
        SPIndex = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),1:2));
        MotionPool = zeros(size(SPIndex,1),2);
        %[gx gy] = gradient(vx{i});
        %temp1 = sqrt(gx.^2+gy.^2);
        %[gx gy] = gradient(vy{i});
        %temp2 = sqrt(gx.^2+gy.^2);
        for k=1:size(SPIndex,1)
           MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2))];
           %MotionPool(k,:) = [MotionContrast(SPIndex(k,1),SPIndex(k,2))];
           %MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2)) temp1(SPIndex(k,1),SPIndex(k,2)) temp2(SPIndex(k,1),SPIndex(k,2))];
           %MotionPool(k,:) = [temp1(SPIndex(k,1),SPIndex(k,2)) temp2(SPIndex(k,1),SPIndex(k,2))];
           %MotionPool(k,:) = [MotionLength{i}(SPIndex(k,1),SPIndex(k,2)) MotionAngle{i}(SPIndex(k,1),SPIndex(k,2))];
        end
        motion{i}{j} = mean((MotionPool));
        
        location{i}{j} = [SP{i}.MiddlePoint(j,1)/W SP{i}.MiddlePoint(j,2)/H];
        
        if(SP{i}.ClusteringPixelNum(1,j)==0)% prevent NaN
            motion{i}{j} = zeros(1,2);
            color{i}{j} = zeros(1,3);
        end
    end
end



