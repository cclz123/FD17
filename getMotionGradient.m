%返回结果为大于Threshold的坐标位置
function [result] = getMotionGradient(vx,vy,I,rate)

DownSamplingSize = 100;
for FrameIndex=1:size(vx,2)
    [gx gy] = gradient(vx{FrameIndex});
    temp1 = sqrt(gx.^2+gy.^2);
    [gx gy] = gradient(vy{FrameIndex});
    temp2 = sqrt(gx.^2+gy.^2);
    MotionGradientMask = temp1+temp2;
    MotionGradientMask = imresize(MotionGradientMask,[DownSamplingSize DownSamplingSize]);%下采样
    %{
    temp4 = zeros(DownSamplingSize,DownSamplingSize);
    meanValue = mean(mean(temp3));
    for i=1:DownSamplingSize
        for j=1:DownSamplingSize
            if(temp3(i,j)>meanValue*2)
                temp4(i,j) = 1;
            end
        end
    end
    %}
    %计算图像的梯度
    tempI = imresize(I{FrameIndex},[DownSamplingSize DownSamplingSize]);%下采样
    [gx gy] = gradient(tempI);
    GradientMask = sqrt(gx.^2+gy.^2);
    CannyEdgeMask = edge(tempI(:,:,1),'canny');
    [CSGMask, gb_thin_CS, gb_CS, orC, edgeImage, edgeComponents] = Gb_CSG(tempI);
    %FusionedMask = sum(CannyEdgeMask,3).*MotionGradientMask;
    FusionedMask = CSGMask.*MotionGradientMask;
    %FusionedMask = sum(GradientMask,3).*MotionGradientMask;
    meanValue = mean(mean(FusionedMask));
    FinalMask = zeros(DownSamplingSize,DownSamplingSize);
    for i=1:DownSamplingSize
        for j=1:DownSamplingSize
            if(FusionedMask(i,j)>meanValue*rate)
                FinalMask(i,j) = 1;
            end
        end
    end
    [XIndex YIndex] = find(FinalMask==1);
    pool = [XIndex YIndex];%pool为所有高gradient的点位置坐标
    pool = pool'.*3;%由于前面进行了下采样，需要进行放大补偿
    result{FrameIndex} = pool;
end
%figure;imshow(temp3,[]);

