function [x, residuals, x_history, i, total_runtime, res_full, varargout] = wrk(A,b,x0,x_true,eta,t,tol,W,p,option_q,capture_diagnostics)
% Weighted randomized Kaczmarz Algorithm with weighting function and
% sampling distribution function, fixed stepsize.
%
% INPUT:
% A: input matrix
% b: output vector
% x0: initial guess for solution
% t: number of iterations
% eta: step size
% W: function that outputs the mxm weighting matrix from the residual
% p: function that outputs the m-dim probability vector from W and A
% option_q: optional quantile fraction
% capture_diagnostics: logical, true = capture s_plot/inlier_mask/sigma_plot
%                      from W's varargout (requires W to support 4 outputs).
%                      false (default) = single-output W call, no diagnostics.
%
% OUTPUT:
% x: solution to linear system
% residuals: residual norms over all iterations
% x_history: approximation error over all iterations
% i: index of last iteration
% varargout (capture_diagnostics=false): {1} = sel_ind (picked row indices)
% varargout (capture_diagnostics=true):  {1}=s_plot, {2}=inlier_mask,
%                                        {3}=sigma_plot, {4}=sel_ind

arguments
    A (:,:) double
    b (:,1) double
    x0 (:,1) double
    x_true (:,1) double
    eta (1,1) double
    t (1,1) int64
    tol (1,1) double
    W (1,1) function_handle
    p (1,1) function_handle
    option_q (1,1) double   = 1.0
    capture_diagnostics (1,1) logical = false
end

[m,~] = size(A);
x = x0; % set initial guess
residuals = []; % initial residual
x_history = [];
total_runtime = 0;
res_full = [];

inlier_mask_history = false(m, t);

for i = 1:t
    tic
    r = A*x - b; % construct iteration residual vector
    if ~capture_diagnostics
    [Wmat] = W(r,option_q);
    else
    [Wmat, s_plot(i), inlier_mask_history(:,i), sigma_plot(i)] = W(r,option_q);
    end
    pvec = p(Wmat,A); % construct iteration sampling weights

    ij = randsample(m,1,true,pvec); % get random sample row index
    sel_ind(i) = ij;
    wij = Wmat(ij,:); % get randomly sampled row of A

    x = x - eta*(wij*r)/(norm(wij*A)^2)*(wij*A)'; % Kaczmarz step
   
    total_runtime = total_runtime + toc;

    residuals = [residuals norm(r)];
    x_history = [x_history norm(x-x_true)];
    res_full = [res_full r];

    if x_history(:,end) < tol 
        break
    end
end
if capture_diagnostics
    varargout{1} = s_plot;
    varargout{2} = inlier_mask_history(:, 1:i);
    varargout{3} = sigma_plot;
    varargout{4} = sel_ind;
else
    varargout{1} = sel_ind;
end
end