function [W, varargout] = fChoice5W(r, n)
% f_choice5 weighting: exp(-r_i^2 / sigma_MAD^2)
% sigma_MAD = MAD(r) / 0.6745
% INPUT:  r (residual vector), n (unused)
% OUTPUT: W (diagonal weight matrix), s=NaN, inlier_mask=all true, sigma=sigma_MAD
eps_floor = 1e-12;
mad   = max(median(abs(r - median(r))), eps_floor);
sigma = mad / 0.6745;
W_vec = exp(-min((r / sigma).^2, 745));
if sum(W_vec) <= eps_floor || ~all(isfinite(W_vec))
    W_vec = ones(size(r));
end
W = diag(W_vec / max(W_vec));
varargout{1} = NaN;
varargout{2} = true(size(r));
varargout{3} = sigma;
end
