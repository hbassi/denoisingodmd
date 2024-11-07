function omega = mp(data,dt,tol)

%% defaults
if nargin < 3, tol = 1e-6; end

%% Hankel matrix
k = length(data);
X = make_hankel(data,floor(k/3)+1,ceil(2/3*k));

%% svd
[~,S,V] = svd(X,'econ');

%% rank truncation
r = sum(diag(S) > tol*S(1));
V = V(:,1:r);
V1 = V(1:end-1,:);
V2 = V(2:end,:);

%% mp
Atilde = pinv(V2)*V1;
mu = eig(Atilde);
omega = log(mu)/dt;

end
