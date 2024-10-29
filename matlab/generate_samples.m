function [H,S] = generate_samples(E,psi0,dt,nb)

%% defaults
if nargin < 2; dt = 1; end
if nargin < 3; nb = 100; end

%% dimensions
n = length(psi0);
assert(length(E) == n);

%% samples
H = zeros(nb,1);
S = zeros(nb,1);
for j = 0:nb-1
    H(j+1) = sum(E.*abs(psi0).^2.*exp(-1i*E*j*dt));
    S(j+1) = sum(abs(psi0).^2.*exp(-1i*E*j*dt));
end

end
