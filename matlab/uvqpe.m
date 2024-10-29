function omega = uvqpe(data,dt,tol)

%% defaults
if nargin < 3, tol = 1e-6; end

%% Toeplitz matrices
if length(data) == 2
    H = data(2);
else
    H = toeplitz([data(2);data(1);conj(data(2:end-2))],data(2:end));
end
S = toeplitz(data(1:end-1));

%% rank truncation
[V,d] = eig(S,'vector');
V = V(:,end:-1:1);
d = d(end:-1:1);
r = sum(d > tol*d(1));
V = V(:,1:r);

%% eigenvalues
Ht = V'*H*V;
St = diag(d(1:r));
mu = eig(Ht,St);
omega = log(mu)/dt;

end
