function [predLabel, predValue, accRate] = svmTest(xTs, yTs, svmMdl, para)
% SVM test for weighted additive kernels
% Input:
%   xTs: test data, each row is a sample
%   yTs: label for test data
%   svmMdl: returned SVM model
%    -alpha: optimal dual variables for SVM
%    -bias: bias for SVM
%    -weight: weight for each dimension of data
%    -svIdx: index for support vector
%    -xTrSV: support vectors
%    -yTrSV: labels of support vectors
%    -totalSV: number of support vectors
%    -activeFeatIdx: weights of active features
%   para: SVM related parameters
% Output:
%   predLabel: predicted label
%   predValue: predicted value
%   accuRate: statistic for testing accuracy
%
% Ji Zhao
% zhaoji84@gmail.com
% 11/19/2012
%
% Reference
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

alpha    = svmMdl.alpha;
bias     = svmMdl.bias;
coeffVec = svmMdl.weight;
svIdx    = svmMdl.svIdx;
xTr      = svmMdl.xTrSV;
yTr      = svmMdl.yTrSV;
alpha = alpha(svIdx);

n = size(xTs, 1);
kmatrix = covAdditve(xTr, xTs, coeffVec, para);
c = alpha.*yTr;
predValue = c'*kmatrix + bias;
predLabel = sign(predValue);
predLabel = predLabel(:);

if ~isempty(yTs) && (numel(yTs)==n)
    idxPos = find(yTs==1);
    nPos = numel(idxPos);
    corrPos = numel(find(predLabel(idxPos)==1));
    idxNeg = find(yTs==-1);
    nNeg = numel(idxNeg);
    corrNeg = numel(find(predLabel(idxNeg)==-1));
    accRate = struct('numPos', nPos, 'numNeg', nNeg, 'corrPos', corrPos, 'corrNeg', corrNeg);
else
    accRate = [];
end
