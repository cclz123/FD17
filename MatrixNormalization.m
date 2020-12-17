function result = MatrixNormalization(M)

maxValue = max(max(M));
minValue = min(min(M));
if(maxValue-minValue~=0)
    M = (M-minValue)/(maxValue-minValue);
end
result = M;