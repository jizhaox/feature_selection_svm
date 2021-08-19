function [alpha, b, p, svIdx, activeFeatIdxs, activeFeatWeights, multiplier] = ipoptMKL(xTr, yTr, binNmlz, p0, para)
% feature selection by normalized margin SVM using Ipopt and SVM solver.
% If all elements in binNmlz are the same, this function is SimpleMKL with
% additive kernel.
% Input:
%	xTr     - training data (n * M)
%	yTr     - vector of labels (n * 1)
%   binNmlz - normalization factors for each bin
%   p0      - initial value for weight vector p
%   para    - SVM related parameters
% Outputs: 
%	alpha - Multipliers for SVM 
%	b     - SVM bias
%	p     - weights for each bin (M * 1)
%   svIdx - indices for support vectors
%   activeFeatIdxs - index of active features (chosen, corresponding to
%       non-zero weights)
%   activeFeatWeights - weights of active features
%   multiplier - lagrange multiplier
%
% You need to install ipopt toolbox and an SVM solver.
% This file was modified based on Peter Vincent Gehler's simpleMKL file.
%   see http://people.kyb.tuebingen.mpg.de/pgehler/ikl-webpage/
%
% Ji Zhao
% zhaoji84@gmail.com
% 11/15/2012
%
% Reference
% [1] Ji Zhao, Liantao Wang, Ricardo Cabral, and Fernando De la Torre. 
%     Feature and Region Selection for Visual Learning. ArXiv: 1407.5245, 2014.

weight_thresh = para.weight_thresh;

% for the Ipopt solver
global stored_p stored_f stored_alpha
global objective_counter retrain_counter gradient_counter
global stored_Jocob

% 0 <= p <= Infty
lb = zeros(size(p0));
ub = Inf(size(p0));
% (binNmlz'*p) should be a fixed value. In paper [1], we set (binNmle'*p) as 1
% For numerical stability we set (binNmlz'*p) as sum(binNmlz), to avoid p being too small.
% These two optimization problems are equivalent by using different
% regularization coefficient C.
sumAbyP = binNmlz'*p0;
lbc = sumAbyP;
ubc = sumAbyP;

% sometimes Ipopt dies, restarting with slightly different p's
% solves this problem and we do this at max 5 times
nof_max_tries = 5; 
nof_tries = 0;
while nof_tries<nof_max_tries
    try
        nof_tries = nof_tries+1;
        objective_counter = 0;
        retrain_counter = 0;
        gradient_counter = 0;

        stored_f = Inf;
        stored_p = Inf(size(p0));
        stored_alpha = [];
        stored_Jocob = binNmlz;

         % SIMPLEMKL
        options.lb = lb;  % Lower bound on the variables.
        options.ub = ub;  % Upper bound on the variables.
        options.cl = lbc; % Lower bounds on constraints.
        options.cu = ubc; % Upper bounds on constraints.
        % The callback functions.
        funcs.objective         = @(p) compute_objective(p,yTr,xTr,para);
        funcs.gradient          = @(p) compute_gradient(p,yTr,xTr,para);
        funcs.constraints       = @(p) compute_constraints(p);
        funcs.jacobian          = @(p) compute_jacobian(p);
        funcs.jacobianstructure = @( ) compute_jacobian_structure();
        %funcs.iterfunc         = @callback; %??
        % Set the IPOPT options.
        options.ipopt.jac_c_constant        = 'yes';
        options.ipopt.hessian_approximation = 'limited-memory';
        options.ipopt.mu_strategy           = 'adaptive';
        options.ipopt.max_iter              = para.ipoptMaxIter;
        %options.ipopt.print_level           = 1;

        % Run IPOPT
        [p,multiplier] = ipopt(p0,funcs,options);
        break;
    catch
        % just add a bit of noise to p and restart
        p0 = p0 + 1e-6*randn(size(p0));
        p0 = p0 * (lbc / (binNmlz'*p0));
        fprintf('ipoptMKL: IpOpt died, retrying after small perturbation\n');
    end
end
    
% compute the final SVM parameters 
alpha = stored_alpha;
p(p<weight_thresh*max(p)) = 0;
activeFeatIdxs = find(p>0);
p = p * (sumAbyP / (binNmlz'*p));
activeFeatWeights = p(activeFeatIdxs);

kmatrix = covAdditve(xTr, [], p, para);
[alpha, ~, b, svIdx] = svmSolvers(kmatrix, yTr, alpha, para);

fprintf('MKL: calls F:%d,',objective_counter);
fprintf('SVM:%d,',retrain_counter);
fprintf('dF:%d\n',gradient_counter);


%
% OBJECTIVE FUNCTION
%
function f = compute_objective(p, yTr, xTr, para)
global stored_p stored_f stored_alpha
global objective_counter retrain_counter

objective_counter = objective_counter + 1;
if all(stored_p == p) && ~isinf(stored_f)
    f = stored_f;
    return;
end

retrain_counter = retrain_counter + 1;
alpha = stored_alpha;
kmatrix = covAdditve(xTr, [], p, para);
[alpha, objDual] = svmSolvers(kmatrix, yTr, alpha, para);
stored_p = p;
stored_f = objDual;
stored_alpha = alpha;
f = objDual;


%
% GRADIENT OF OBJECTIVE FUNCTION
%
function df = compute_gradient(p, yTr, xTr, para)
global stored_p stored_f stored_alpha
global retrain_counter gradient_counter;
gradient_counter = gradient_counter +1;
retrain_counter = retrain_counter + 1;

% if all(stored_p == p)
%     df = - stored_norm;
%     return;
% end

alpha = stored_alpha;
kmatrix = covAdditve(xTr, [], p, para);
[alpha, objDual, ~, svIdx] = svmSolvers(kmatrix, yTr, alpha, para);
g = gradientForKernelWeight(alpha, svIdx, xTr, yTr, para);
stored_p = p;
stored_alpha = alpha;
stored_f = objDual;
df = g;

%
% CONSTRAINT sum(p) = constant
%
function f = compute_constraints(p)
global stored_Jocob
tmp = stored_Jocob;
f = tmp'*p;

%
% JACOBIAN
%
function J = compute_jacobian(p)
global stored_Jocob
tmp = stored_Jocob;
tmp = tmp(:);
d = numel(tmp);
J = sparse(ones(d,1), (1:d)', tmp, 1, d);

%
% JACOBIAN STRUCTURE
%
function J = compute_jacobian_structure()
global stored_Jocob
J = sparse(ones([1 numel(stored_Jocob)]));
