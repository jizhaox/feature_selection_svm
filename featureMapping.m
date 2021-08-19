function [psiData] = featureMapping(Data, kernelType)
% calculate  finite-dimensional feature mapping for additve kernels
% Input:
%   Data: data matrix, each column is a sample
%   kernelType: five types of kernels are supported: including chi-squared, 
%     histogram intersection, Jensen-Shannon, linear, and Hellinger [1]
% Output:
%   psiData: finite-dimensional feature mapping
%
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 12/23/2012
%
% Reference:
% [1] A. Vedaldi and A. Zisserman. Efficient Additive Kernels via Explicit
%     Feature Maps. IEEE Trans. PAMI, 2012.

if (strcmp(kernelType, 'kchi2') || strcmp(kernelType, 'kinters') || strcmp(kernelType, 'kjs'))
    psiData = vl_homkermap(Data, 1, kernelType);
    % re-organize the order of dimensions, making it more convient for CVX solver
    psiData = [psiData(1:3:end,:); psiData(2:3:end,:); psiData(3:3:end,:)];
elseif (strcmp(kernelType, 'Linear'))
    psiData = Data;
elseif (strcmp(kernelType, 'Hellinger'))
    psiData = sqrt(Data);
else
    error('unknown kernel type in function featureMapping!')
end


