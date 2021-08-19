function [v, b, p, activeFeatIdxs, activeFeatWeights, objVal] = initByFeatMapping(NData, label, binNmlz, coeffVector, para)
% Learning the weights for normalized SVM with approximated additive kernels
% This function tries to balance the size of two classes. 
% This function also balances between C and # of constraints; thus
% even the number of samples increase, the C can be kept constant. 
%
% Inputs:
%   NData: data matrix of size d*n, d: # features, n: # training data.
%   label: corresponding labels of training data, this is column vector.
%       the entries must be either 1 or -1.
%   para.regLambda: the parameter C in SVM.
% Outputs:
%   v: weight vector for SVM.
%   b: the bias of the SVM.
%   p: weights for dimensions.
%   activeFeatIdxs: index of active features (chosen, corresponding to
%       non-zero weights)
%   activeFeatWeights: weights of active features
%   objVal: optimal value of objective funtion
% 
% Ji Zhao
% zhaoji84@gmail.com
% 02/18/2013
%
% Reference
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

wPos = para.weightPosSamp;
wNeg = para.weightNegSamp;
C = para.regLambda;
weight_thresh = para. weight_thresh;

% (binNmlz'*p) should be a fixed value. In paper [1], we set (binNmle'*p) as 1
% For numerical stability we set (binNmlz'*p) as sum(binNmlz), to avoid p being too small.
% These two optimization problems are equivalent by using different
% regularization coefficient C.
sumAbyP = binNmlz'*coeffVector;


NData = double(NData);
label = double(label);
%Data = NData;
[d, n] = size(NData);

% weights for constraints
constrW = zeros(n, 1);
constrW(label == 1)  = wPos;
constrW(label == -1) = wNeg;
    
str = para.kernelType;
Data = featureMapping(NData, str);
LabelData = bsxfun(@times, Data, label');
if (strcmp(str, 'kchi2') || strcmp(str, 'kinters') || strcmp(str, 'kjs'))
    [v, b, p, objVal] = optCvxEngineAdditive(LabelData, label, binNmlz, constrW, C, sumAbyP);
elseif (strcmp(str, 'Linear') || strcmp(str, 'Hellinger'))
    [v, b, p, objVal] = optCvxEngineLinear(LabelData, label, binNmlz, constrW, C, sumAbyP);
else
    error('unknow kernel type in function initByFeatMapping!');
end

p(p<weight_thresh*max(p)) = 0;
activeFeatIdxs = find(p>0);
p = p * (sumAbyP / (binNmlz'*p));
activeFeatWeights = p(activeFeatIdxs);

%% display results
rsltDsip = 1;
if rsltDsip
    svmScore = v'*Data + b;
    idxPos = find(label==1);
    nPos = numel(idxPos);
    corrPos = numel(find(svmScore(idxPos)>0));
    idxNeg = find(label==-1);
    nNeg = numel(idxNeg);
    corrNeg = numel(find(svmScore(idxNeg)<0));
    fprintf('  true positive rate:%4.2f%%\n', corrPos/nPos*100);
    fprintf('  true negative rate: %4.2f%%\n', corrNeg/nNeg*100);
    fprintf('  Number of non-zero weights: %d\n', numel(activeFeatWeights));
end;


function [w, b, p, objVal] = optCvxEngineAdditive(LabelData, label, a, constrW, C, sumAbyP)
% optimization using quadratic over linear
[d, n] = size(LabelData);
d2 = d/3;
% Note: we safely remove the lower bound constraints for p
% because quad_over_lin(v, p) is inf if p<=0
% see http://cvxr.com/cvx/doc/basics.html
% https://see.stanford.edu/materials/lsocoee364a/cvx_usrguide.pdf
cvx_precision high;
cvx_begin
    cvx_solver SDPT3
    variable w(d);
    variable p(d2);
    variable xi(n);
    variable b;
    minimize(0.5*sum(quad_over_lin(w, [p; p; p], 2)) + C*(constrW'*xi));
    subject to
        LabelData'*w + b*label + xi >= ones(n, 1);
        a'*p == sumAbyP;
        xi >= zeros(n,1);
cvx_end
if strcmp(cvx_status, 'Failed')
    error('CVX failed ...');
end;
objVal = cvx_optval;
 

function [w, b, p, objVal] = optCvxEngineLinear(LabelData, label, a, constrW, C, sumAbyP)
% optimization using quadratic over linear
[d, n] = size(LabelData);
% Note: we safely remove the lower bound constraints for p
% because quad_over_lin(v, p) is inf if p<=0
% see http://cvxr.com/cvx/doc/basics.html
% https://see.stanford.edu/materials/lsocoee364a/cvx_usrguide.pdf
cvx_precision high;
cvx_begin
    cvx_solver SDPT3
    variable w(d);
    variable p(d);
    variable xi(n);
    variable b;
    minimize(0.5*sum(quad_over_lin(w, p, 2)) + C*(constrW'*xi));
    subject to
        LabelData'*w + b*label + xi >= ones(n, 1);
        a'*p == sumAbyP;
        xi >= zeros(n,1);
cvx_end
if strcmp(cvx_status, 'Failed')
    error('CVX failed ...');
end;
objVal = cvx_optval;
