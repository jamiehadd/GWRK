function [W, varargout] = fChoice1W(r, n)
% f_choice1 weighting: exp(-r_i^2 / median(|r|)^2)
% INPUT:  r (residual vector), n (unused)
% OUTPUT: W (diagonal weight matrix), s=NaN, inlier_mask=all true, sigma=median(|r|)
eps_floor = 1e-12;
abs_r  = abs(r);
sigma  = max(median(abs_r), eps_floor);
W_vec  = exp(-min((abs_r / sigma).^2, 745));
if sum(W_vec) <= eps_floor || ~all(isfinite(W_vec))
    W_vec = ones(size(r));
end
W = diag(W_vec / max(W_vec));
varargout{1} = NaN;
varargout{2} = true(size(r));
varargout{3} = sigma;
end
