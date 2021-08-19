function kappa = covKappa(x,z,para, useFeatMap)
% kernel matrix for 1-dimensional data
% Input:
%   x: columnn vector, each element is 1-dimensional data
%   z: column vector or empty
%   para: SVM related parameters
%   para.kernelType: 6 kinds of additve kernels are supported, including 
%     chi-squared, histogram intersection, Jensen-Shannon, linear, 
%     Hellinger and power mean [2][3]
%   useFeatMap: whether enable feature mapping approximation, true or false
% Output:
%   kappa: kernel matrix
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

if nargin<4
    useFeatMap = false;
end

nx = numel(x);
xeqz = isempty(z); % determine mode
kernelType = para.kernelType;
switch kernelType
    case 'kchi2' % chi squared
        if (~useFeatMap)
            if (xeqz)
                kappa = 2*(x*x')./(bsxfun(@plus, x, x')+eps);
            else
                kappa = 2*x*z'./(bsxfun(@plus, x, z')+eps);
            end
        else
            if (xeqz)
                psix = vl_homkermap(x', 1, 'kchi2');
                kappa = psix'*psix;
            else
                psix = vl_homkermap(x', 1, 'kchi2');
                psiy = vl_homkermap(y', 1, 'kchi2');
                kappa = psix'*psiy;
            end
        end
    case 'kinters' % histogram intersection
        if (xeqz)
            kappa = bsxfun(@min, x, x');
        else
            kappa = bsxfun(@min, x, z');
        end
    case 'kjs' % Jensen-Shannon
        if (xeqz)
            k1 = repmat(x, [1 nx]);
            k2 = repmat(x', [nx 1]);
            kappa = ( k1.*log2(1+k2./k1) + k2.*log2(1+k1./k2) )/2;
        else
            k1 = repmat(x,[1 numel(z)]);
            k2 = repmat(z', [nx 1]);
            kappa = ( k1.*log2(1+k2./k1) + k2.*log2(1+k1./k2) )/2;
        end
    case 'Linear'
        if (xeqz)
            kappa = x*x';
        else
            kappa = x*z';
        end
    case 'Hellinger' % Hellinger
        if (xeqz)
            kappa = sqrt( bsxfun(@times, x, x') );
        else
            kappa = sqrt( bsxfun(@times, x, z') );
        end
    case 'PowerMean'
        % p \in (-inf, 0]
        try
            p = para.p;
        catch
            p = -1;
        end
        if (xeqz)
            kappa = ((repmat(x,[1 nx]).^p + repmat(x', [nx 1]).^p)/2).^(1/p);
        else
            kappa = ((repmat(x,[1 numel(z)]).^p + repmat(z', [nx 1]).^p)/2).^(1/p);
        end
    otherwise
        error('unknown kernel type!');
end
