% feature selection by normalized margin SVM
% using additive kernels
%
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 10/22/2012
%
% Reference:
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

clear;
%% parameters
% 5 types kernels are supported: 
%   chi-squared, linear, histogram intersection, Jensen-Shannon, Hellinger.
kernelSet = {'kchi2', 'Linear', 'kinters', 'kjs', 'Hellinger'};
% kernel type
para.kernelType = kernelSet{1};
% parameter C in SVM
para.regLambda = 30;
% whether use fast approximate solution for initialization
para.initByFeatMap = true;
% maximum iteration in IPOPT solver
para.ipoptMaxIter = 50;
% threshold to set small weights as zeros
para.weight_thresh = 1e-2;

%% install libSVM, IPOPT, CVX and VLfeat toolboxs
% copy "svmtrain" file in libSVM to current path
% because Matlab has a built-in function which has the same name
if (~exist('libsvmread', 'file'))
    path(path, './3rdParty/libsvm-3.20');
end
if (~exist('ipopt', 'file'))
    path(path, './3rdParty/Ipopt-3.11.8-linux64mac64win32win64-matlabmexfiles')
end
if (~exist('cvx_setup', 'file'))
    path(path, './3rdParty/cvxw-32/cvx');
    cvx_setup();
end
if (~exist('vl_homkermap', 'file'))
    path(path, './3rdParty/vlfeat-0.9.20/toolbox');
    vl_setup();
end
%% load and prepare data
pathData = 'data\data_imgcls_375_100';
[xTr, yTr, xTs, yTs] = prepareData(pathData, 2);

%% SVM with feature selection
[svmMdl, svmMdlAppro] = featureSelectionAddKernel(xTr, yTr, para);

%% SVM test with approximate solution
if ~isempty(svmMdlAppro)
    [~, svmTrScore] = svmTestAppro(xTr', yTr, svmMdlAppro, para.kernelType);
    [~, svmTsScore] = svmTestAppro(xTs', yTs, svmMdlAppro, para.kernelType);
end

%% SVM test with exact solution
[predTrL, predTrScore, accTr] = svmTest(xTr, yTr, svmMdl, para);
[predTsL, predTsScore, accTs] = svmTest(xTs, yTs, svmMdl, para);

%% output and visualization
fprintf('Accuracy for positive/negative training data is %4.2f%% and %4.2f%%\n',...
    accTr.corrPos/accTr.numPos*100, accTr.corrNeg/accTr.numNeg*100)
fprintf('Accuracy for postive/negative test data is %4.2f%% and %4.2f%%\n',...
    accTs.corrPos/accTs.numPos*100, accTs.corrNeg/accTs.numNeg*100)
fprintf('Number of non-zero weights: %d\n', numel(svmMdl.activeFeatIdx));

% visualization of feature weights
figure, stem(svmMdl.weight, 'k.'), 
if ~isempty(svmMdlAppro)
    hold on; stem(svmMdlAppro.weight, 'b.'), legend('exact solution', 'approximate solution')
end
xlabel('feature index'), ylabel('weight'), title('feature weights for normalized margin SVM')
axis tight;

% comparision of exact solution and approximate solution
figure, stem(predTrScore, 'k.'), 
if ~isempty(svmMdlAppro)
    hold on, stem(svmTrScore, 'b.'), legend('exact solution', 'approximate solution')
end
xlabel('feature index'), ylabel('SVM score'), title('SVM score for training set')
axis tight
figure, stem(predTsScore, 'k.'), 
if ~isempty(svmMdlAppro)
    hold on, stem(svmTsScore, 'b.'), legend('exact solution', 'approximate solution')
end
xlabel('feature index'), ylabel('SVM score'), title('SVM score for test set')
axis tight

