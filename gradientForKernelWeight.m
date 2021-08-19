function g = gradientForKernelWeight(alpha, activeIdx, xTr, yTr, para)
% calculate gradient with alpha in SVM dual domain
% feature selection by normalized margin SVM using additive kernels
% Input:
%   alpha: optimal variables in SVM dual domain
%   activeIdx: index for bins that have non-zero weights
%   xTr: training data
%   yTr: label for training data
%   para: SVM related parameters
% Output:
%   g: gradient with alpha
%
% Ji Zhao
% zhaoji84@gmail.com
% 10/29/2012
%
% Reference:
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

nTr = size(xTr, 1);
dim = size(xTr, 2);
g = zeros(dim, 1);
nx = numel(activeIdx);

% squeeze non-support vectors
if (nx < nTr)
    alpha = alpha(activeIdx);
    xTr = xTr(activeIdx, :);
    yTr = yTr(activeIdx, :);
end

alpha = alpha .* yTr;
for i = 1:dim
    kappa = zeros(nx, nx);
    xFeat = xTr(:,i);
    % squeeze bins that are zero
    xIdx = find(xFeat>0);
    if numel(xIdx)~=nx
        xFeat = xFeat(xIdx);
    end
    kappa(xIdx, xIdx) = covKappa(xFeat, [], para);
    g(i) = - alpha' * kappa * alpha / 2;
end
