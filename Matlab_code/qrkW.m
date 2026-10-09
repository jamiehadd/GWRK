function W = qrkW(r,q,~)
% Weight matrix function that makes weighted RK specialize to QRK.
%
% INPUT:
% r: residual vector
%
% OUTPUT: 
% W: weight matrix

m = size(r,1);
W = diag(abs(r) <= quantile(abs(r),q));