function p = l2erk1p(W,~)
% Sampling weight function that makes weighted RK specialize to L2ERK1.
%
% INPUT:
% W: weight matrix
%
% OUTPUT: 
% p: sampling weights

p = diag(W)/trace(W);