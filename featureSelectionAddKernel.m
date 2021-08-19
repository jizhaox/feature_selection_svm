function [svmMdl, svmMdlAppro] = featureSelectionAddKernel(xTr, yTr, para)
% feature selection for normalized margin SVM
% Input:
%   xTr: training data, each row is a sample
%   yTr: label for traing data, +1 or -1 only
%   para: SVM related parameters
% Output:
%   svmMdl: returned SVM model
%    -alpha: optimal dual variables for SVM
%    -bias: bias for SVM
%    -coeffVector: weight for each dimension of data
%    -svIdx: index for support vector
%    -xTrSV: support vectors
%    -yTrSV: labels of support vectors
%    -totalSV: number of support vectors
%    -activeFeatIdx: weights of active features
%
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 11/01/2012
%
% Reference:
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

nTrPos = numel(find(yTr==1));
nTrNeg = numel(find(yTr==-1));
para.weightPosSamp = nTrNeg/nTrPos;
para.weightNegSamp = 1;
dim = size(xTr, 2);
xTr_backup = xTr;
%% calculate the normalization factors for each bin
binNmlz = binNormalize(xTr, yTr, para);
%binNmlzFeatMap = binNormalize(xTr, yTr, para, true);
% remove dimensions that have zero normalization factors
idxSub = find(binNmlz>0);
dimSub = numel(idxSub);
if (dimSub==0)
    error('data error!');
elseif (dimSub < dim)
    binNmlz = binNmlz(idxSub);
    xTr = xTr(:, idxSub);
end
coeffVector = ones(dimSub, 1);

%% initialize by feature mapping
if para.initByFeatMap
    sumAbyP = binNmlz'*coeffVector;
    tryNum = 3;
    for ii = 1:tryNum;
        try
            tic
            [v, b, p] =  initByFeatMapping(xTr', yTr, binNmlz, coeffVector, para);
            toc
            coeffVector = p;
            break;
        catch
            %just add a bit of noise to w and restart
            coeffVector = coeffVector + 1e-6*randn(size(coeffVector));
            coeffVector = coeffVector * (sumAbyP / (binNmlz'*coeffVector));
            fprintf('CVX died, retrying after small perturbation\n');
        end
    end
end
try
    n = numel(v)/numel(p);
    tmp = zeros(dim, 1);
    tmp(idxSub) = p;
    p = tmp;
    tmp = zeros( n*dim, 1);
    idt = [];
    for i = 0:n-1
        idt = [idt; idxSub(:)+i*dim];
    end
    tmp(idt) = v;
    v = tmp;
    svmMdlAppro = struct('w', v, 'bias', b, 'weight', p);
catch
    svmMdlAppro = [];
end

%% IPopt optimization
tic
[alpha, bias, coeffVector, svIdx, activeFeatIdxs] = ipoptMKL(xTr, yTr, binNmlz, coeffVector, para);
toc
%%
tmp = zeros(dim, 1);
tmp(idxSub) = coeffVector;
coeffVector = tmp;

tmp = zeros(dim, 1);
tmp(idxSub) = binNmlz;
binNmlz = tmp;

%% format conversion
svmMdl = struct('alpha', alpha, 'bias', bias, 'weight', coeffVector, 'svIdx', ...
    svIdx, 'xTrSV', xTr_backup(svIdx,:), 'yTrSV', yTr(svIdx), 'totalSV', numel(svIdx), ...
    'activeFeatIdx', activeFeatIdxs);



