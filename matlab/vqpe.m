function lam = vqpe(dataH,dataS,tol)

%% defaults
if nargin < 3, tol = 1e-6; end

%% Toeplitz matrices
H = toeplitz(dataH);
S = toeplitz(dataS);

%% rank truncation
[V,d] = eig(S,'vector');
V = V(:,end:-1:1);
d = d(end:-1:1);
r = sum(d > tol*d(1));
V = V(:,1:r);

%% eigenvalues
Ht = V'*H*V; Ht = (Ht + Ht')/2;  % make it Hermitian
St = diag(d(1:r));
lam = eig(Ht,St);

end
