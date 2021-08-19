function binNmlz = binNormalize(xTr, yTr, para, useFeatMap)
% calculate the normalization factor for normalized margin SVM
% It is the summation of pairwise distance for positive & negative data
% Input:
%   xTr: training data, each row is a sample
%   yTr: label for traing data, +1 or -1 only
%   para: SVM related parameters
%   useFeatMap: whether enable feature mapping approximation, true or false
% Output:
%   binNmlz: normalization factors for each bin
%
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 11/01/2012
%
% Reference:
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

if nargin<4
    useFeatMap = false;
end

[nx,d] = size(xTr);
idxPos = find(yTr==1);
idxNeg = find(yTr==-1);

binNmlz = zeros(d, 1);
for k = 1:d
    xFeat = xTr(:,k);
    xIdx = find(xFeat>0);
    if (numel(xIdx)~=nx)
        xFeat = xFeat(xIdx);
    end
    kappa = covKappa(xFeat, [], para, useFeatMap);
    [~, ~, idx1] = intersect(idxPos, xIdx);
    [~, ~, idx2] = intersect(idxNeg, xIdx);
    dg = diag(kappa);
    binNmlz(k) = -sum(sum(kappa(idx1,idx1))) + sum(dg(idx1))*numel(idxPos) ...
                -sum(sum(kappa(idx2,idx2))) + sum(dg(idx2))*numel(idxNeg);
end
binNmlz = binNmlz*2;
%binNmlz2 = covEachBinNaive(xTr, yTr, para);
%max(abs(binNmlz-binNmlz2)./binNmlz)

function [binNmlzNaive, binNmlzFeatMap] = covEachBinNaive(xTr, yTr, para)
[nx, d] = size(xTr);
binNmlzNaive = zeros(d, 1);
binNmlzFeatMap = zeros(d, 1);

%for k = 1:10
for k = 1:d
    a1 = 0;
    a2 = 0;
    x = xTr(:, k);
    kappa1 = covKappa(x, [], para);
    psix = vl_homkermap(x', 1, 'kchi2');
    kappa2 = psix'*psix;
    for i = 1:nx
        for j = 1:nx
            if (yTr(i)==yTr(j))
                a1 = a1 + kappa1(i,i) - 2*kappa1(i,j) + kappa1(j,j);
                a2 = a2 + kappa2(i,i) - 2*kappa2(i,j) + kappa2(j,j);
            end
        end
    end
    binNmlzNaive(k) = a1;
    binNmlzFeatMap(k) = a2;
end
