function p = qrkp(W,~)
% Sampling weight function that makes weighted RK specialize to QRK.
%
% INPUT:
% W: weight matrix
%
% OUTPUT: 
% p: sampling weights

p = diag(W)/trace(W);