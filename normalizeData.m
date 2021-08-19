function data2 = normalizeData(data, optRC, optNorm)
% normalize input data
% Input:
%   data: input data
%   optRC: according to row or column, 'row' or 'col'
%   optNorm: L1 norm or L2 norm, 'L1' or 'L2'
% Output:
%   data2: normalized data
%
% Ji Zhao
% zhaoji84@gmail.com
% 10/22/2012
%

[r, c] = size(data);
if (r<1 || c<1)
    data2 = data;
    return;
end

if(strcmp(optRC, 'row') && strcmp(optNorm, 'L1'))
    s = sum(data, 2);
    s(s==0) = eps;
    s = reshape(s, [r, 1]);
    data2 = bsxfun(@rdivide, data, s);
end
if(strcmp(optRC, 'col') && strcmp(optNorm, 'L1'))
    s = sum(data, 1);
    s(s==0) = eps;
    s = reshape(s, [1, c]);
    data2 = bsxfun(@rdivide, data, s);
end
if(strcmp(optRC, 'row') && strcmp(optNorm, 'L2'))
    data2 = normr(data);
end
if(strcmp(optRC, 'col') && strcmp(optNorm, 'L2'))
    data2 = normc(data);
end
