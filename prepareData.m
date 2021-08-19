function [xTr, yTr, xTs, yTs] = prepareData(pathData, clsIdx)
%   xTr: training data, each row is a sample
%   yTr: label for traing data
%   xTs: test data, each row is a sample
%   yTs: label for test data

if nargin < 2
    clsIdx = 1;
end

load(pathData);
xTr = hist_tr';
xTs = hist_ts';

yTr = zeros(numel(lab_tr), 1);
yTr(lab_tr==clsIdx) = 1;
yTr(lab_tr~=clsIdx) = -1;

yTs = zeros(numel(lab_ts), 1);
yTs(lab_ts==clsIdx) = 1;
yTs(lab_ts~=clsIdx) = -1;

xTr = normalizeData(xTr, 'row', 'L1');
xTs = normalizeData(xTs, 'row', 'L1');
