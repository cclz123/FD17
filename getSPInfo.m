function [SP] = getSPInfo(rgbI,hsvI)

%I = cell2mat(im);

%[l, Am, C] = slic(I, 1000, 40, 1, 'mean');
%SEGMENTS = spdbscan(l, C, Am, 5);
SEGMENTS = vl_slic(single(rgbI),13,0.001);%13 is the best choice
SEGMENTS = SEGMENTS + 1;
SP.SuperPixelNumber = max(max(SEGMENTS));

%出现某些SP的像素个数为0的情况


%组装超像素点数据结构
SP.Clustering = zeros(SP.SuperPixelNumber,2000,5);%每个超像素点所包含的像素个数
SP.ClusteringPixelNum = zeros(1,SP.SuperPixelNumber);%
for i=1:size(rgbI,1)
    for j=1:size(rgbI,2)
        SIndex = SEGMENTS(i,j);
        if(SP.ClusteringPixelNum(1,SIndex)+1<=2000)
            SP.ClusteringPixelNum(1,SIndex) = SP.ClusteringPixelNum(1,SIndex)+1;
            SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),:) = [i j hsvI(i,j,1) hsvI(i,j,2) hsvI(i,j,3)]';%记录全部像素点
        end
    end
end

SP.MiddlePoint = zeros(SP.SuperPixelNumber,5);

InvalidIndex = zeros(1,SP.SuperPixelNumber);

BoundaryIndex = zeros(1,SP.SuperPixelNumber);%用于记录边界点

MarginLock = 8;
ImageSize = size(rgbI,1);

for i=1:SP.SuperPixelNumber
    sum_x = 0;sum_y = 0;sum_r = 0;sum_g = 0;sum_b = 0;
    for j=1:SP.ClusteringPixelNum(1,i)
       XIndex = SP.Clustering(i,j,1);
       YIndex = SP.Clustering(i,j,2);
       sum_x = sum_x + XIndex;
       sum_y = sum_y + YIndex;
       sum_r = sum_r + hsvI(XIndex,YIndex,1);
       sum_g = sum_g + hsvI(XIndex,YIndex,2);
       sum_b = sum_b + hsvI(XIndex,YIndex,3);
    end

    if(SP.ClusteringPixelNum(1,i)~=0)
       SP.MiddlePoint(i,:) = ([sum_x sum_y sum_r sum_g sum_b]./SP.ClusteringPixelNum(1,i))';
       XIndex = sum_x/SP.ClusteringPixelNum(1,i);
       YIndex = sum_y/SP.ClusteringPixelNum(1,i);
       if(XIndex<2*MarginLock || XIndex>ImageSize-2*MarginLock || YIndex<2*MarginLock || YIndex>ImageSize-2*MarginLock)
           SP.BoundaryIndex(1,i) = 0.5;%处于边界上
           if(XIndex<MarginLock || XIndex>ImageSize-MarginLock || YIndex<MarginLock || YIndex>ImageSize-MarginLock)
               SP.BoundaryIndex(1,i) = 0.25;
           end
       else
           SP.BoundaryIndex(1,i) = 1;%处于边界内
       end 
    else
        SP.MiddlePoint(i,:) = ([0 0 0 0 0 ])'; 
        InvalidIndex(i,1) = 1;
    end
    
end

%依据InvalidIndex进行剔除
[value index] = find(InvalidIndex==1);
if(size(index,1)~=0)
SP.SuperPixelNumber = SP.SuperPixelNumber - size(index,1);
SP.MiddlePoint(index,:) = [];
SP.ClusteringPixelNum(:,index) = [];
SP.Clustering(index,:,:) = [];
SP.BoundaryIndex(:,index) = [];
end
%{
gI = rgb2gray(hsvI);
[row,col] = size(gI);
SPMap = zeros(row,col,3);%SPMap-SLIC Result
for i=1:row
    for j=1:col
       SPMap(i,j,1) = SP.MiddlePoint(SEGMENTS(i,j),3);
       SPMap(i,j,2) = SP.MiddlePoint(SEGMENTS(i,j),4);
       SPMap(i,j,3) = SP.MiddlePoint(SEGMENTS(i,j),5);
    end
end
%}
