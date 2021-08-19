function [predLabel, predValue, accRate] = svmTestAppro(xTs, yTs, svmMdlAppro, kernelType)
% SVM test for weighted additive kernels
% Input:
%   xTs: test data, each column is a sample
%   yTs: label for test data
%   svmMdlAppro: returned SVM model with finite-dimensional feature mapping
%    -w: hyperplane parameter for linear SVM
%    -bias: bias for SVM
%    -weight: weight for each dimension of data
%   kernelType: five types of kernels are supported: including chi-squared, 
%     histogram intersection, Jensen-Shannon, linear, and Hellinger [1]
% Output:
%   predLabel: predicted label
%   predValue: predicted value
%   accuRate: statistic for testing accuracy
%
% Ji Zhao
% zhaoji84@gmail.com
% 11/26/2012
%
% Reference
% [1] A. Vedaldi and A. Zisserman. Efficient Additive Kernels via Explicit
%     Feature Maps. IEEE Trans. PAMI, 2012.

w = svmMdlAppro.w;
bias = svmMdlAppro.bias;
n = size(xTs, 2);

Data = featureMapping(xTs, kernelType);
predValue = w'*Data + bias;
predValue = predValue(:);
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
