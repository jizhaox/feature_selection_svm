function [alpha, objDual, b, svIdx] = svmSolvers(K, y, alpha0, para)
% SVM interface for different SVM solvers
% Input:
%   K: kernel matrix
%   y: label for training data
%   alpha0: initial value for variables alpha in dual domain
%   para: SVM related parameters
% Output:
%   alpha: optimal variable in dual domain
%   objDual: optimal value for objective function in dual domain
%   b: bias of SVM
%   svIdx: index for support vectors
%
% Ji Zhao
% zhaoji84@gmail.com
% 10/26/2012
%
% Reference
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

if nargin<4
    wPos = 1;
    wNeg = 1;
    reglambda = 1;
else
    wPos = para.weightPosSamp;
    wNeg = para.weightNegSamp;
    reglambda = para.regLambda;
end
alpha_eps = 1e-3;

y = y(:);
%% SVM solver - libSVM
n = size(K, 1);
K1 = [(1:n)', K];
opts = sprintf('-t 4 -c %g -w1 %g -w-1 % g -q', reglambda, wPos, wNeg);
model = svmtrain(y, K1, opts);
alpha = zeros(n, 1);
alpha(model.SVs) = abs(model.sv_coef);
alpha(alpha<alpha_eps*reglambda) = 0;
c = alpha.*y;
objDual = sum(alpha(:)) - c'*K*c/2;

b = -model.rho;
svIdx = model.SVs;



