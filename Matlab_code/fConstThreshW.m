function [W, varargout] = fConstThreshW(r, threshold)
% Weighted L2E with a FIXED inlier threshold.
% The threshold is set once before the algorithm runs (e.g. median of the
% initial residual) and does not change across iterations.
%
% Within the inlier set, weights follow the L2E formula (same as l2erk2W).
% Outlier rows (|r_i| > threshold) receive zero weight every iteration.
%
% INPUT:
%   r:         current residual vector (changes each iteration)
%   threshold: fixed scalar cutoff — rows with |r_i| <= threshold are inliers
%              Pass via closure: @(r,n) fConstThreshW(r, thresh0)
%              where thresh0 = median(abs(A*x0 - b_tilde)) before the loop.
%
% OUTPUT:
%   W:            m x m diagonal weight matrix
%   varargout{1}: threshold (constant, echoed for diagnostics)
%   varargout{2}: inlier_mask (logical, may change as residuals evolve)
%   varargout{3}: sigma (MAD-based scale of current residual)

eps_floor = 1e-12;
abs_r     = abs(r);

% Fixed inlier mask: determined by the constant threshold, not current scale
inlier_mask = abs_r <= threshold;

% MAD-based scale from the current residual (for the exp weights)
mad   = median(abs(r - median(r)));
mad   = max(mad, eps_floor);
tau   = 0.6745 / mad;
sigma = 1 / tau;

% L2E weights on inlier rows only (zero elsewhere)
W_vec = zeros(size(r));
if any(inlier_mask)
    exponent           = (tau / 2) * r(inlier_mask).^2;
    exponent           = exponent - max(exponent);   % log-sum-exp shift
    W_vec(inlier_mask) = exp(exponent);
end

% Fallback: if no inlier survives, treat all rows equally
if sum(W_vec) <= eps_floor || ~all(isfinite(W_vec))
    W_vec       = ones(size(r));
    inlier_mask = true(size(r));
end

W            = diag(W_vec);
varargout{1} = threshold;
varargout{2} = inlier_mask;
varargout{3} = sigma;
end
