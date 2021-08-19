function [kmatrix] = covAdditve(x, z, coeffVector, para)
% calculate kernel matrix for additive kernels
% Input:
%   x: input data, each row is an input pattern
%   z: input data or empty
%   coeffVector: weights for each bin of input data
%   para.kernelType: 6 kinds of additve kernels are supported, including 
%     chi-squared, histogram intersection, Jensen-Shannon, linear, 
%     Hellinger and power mean [2][3]
% Output:
%   kmatrix: kernel matrix
%     if z is empty, return symmetric kernel matrix K(x, x)
%     if x is non-empty, return cross kernel matrix k(x, z)
%
% Ji Zhao@CMU
% zhaoji84@gmail.com
% 02/23/2015
%
% Reference:
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.
% [2] A. Vedaldi and A. Zisserman. Efficient Additive Kernels via Explicit
%     Feature Maps. IEEE Trans. PAMI, 2012.
% [3] Jianxin Wu. Power Mean SVM for Large Scale Visual Classification.
%     CVPR, 2012.

xeqz = isempty(z);
[nx,d] = size(x);
if (xeqz)
    nz = nx;
else
    [nz,d1] = size(z);
    if d~=d1
        error('');
    end
end
if isempty(coeffVector)
    coeffVector = ones(d, 1);
end
kmatrix = zeros(nx,nz);
for i=1:d
    if (coeffVector(i)==0)
        continue;
    end
    xFeat = x(:,i);
    xIdx = find(xFeat>0);
    if isempty(xIdx)
        continue;
    end
    if (numel(xIdx)~=nx)
        xFeat = xFeat(xIdx);
    end
    kappa = zeros(nx, nz);
    if (xeqz) % symmetric matrix Kxx
        kappa(xIdx, xIdx) = covKappa(xFeat, [], para);
    else      % cross covariances Kxz
        zFeat = z(:,i);
        zIdx = find(zFeat>0);
        if isempty(zIdx)
            continue;
        end
        if numel(zIdx)~=nz
            zFeat = zFeat(zIdx);
        end
        kappa(xIdx, zIdx) = covKappa(xFeat, zFeat, para);
    end
    kmatrix = kmatrix + coeffVector(i)*kappa;
end

