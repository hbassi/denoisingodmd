function [omega, sv, bnd] = dmd(data,dt,tol)

%% defaults
if nargin < 3, tol = 1e-6; end

%% Hankel matrix
k = length(data);
X = make_hankel(data,floor(k/2 + 1),ceil(k/2));

%% (shifted) data matrices
X1 = X(:,1:end-1);
X2 = X(:,2:end);


%% svd
[U,S,V] = svd(X1,'econ');

%% rank truncation

r = sum(diag(S) > tol*S(1));

U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

sv = 0;
bnd = -1;
%% dmd
Atilde = U'*X2*V/S;
% Atilde = S^(-1/2)*U'*X2*V*S^(-1/2);
mu = eig(Atilde);
omega = log(mu)/dt;

end
