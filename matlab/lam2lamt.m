function lamt = lam2lamt(lam,lammin,lammax)
%lam2lamt  Convert numbers to [-pi/4,pi/4]
%   Y = lam2lamt(X) converts the matrix X into the interval [-pi/4,pi/4] where
%   min(X) is mapped to -pi/4 and max(X) to pi/4.
%
%   Y = lam2lamt(X,min,max) converts the matrix X where min is mapped to -pi/4
%   and max to pi/4.
%
%   See also lamt2lam.

%% defaults
if nargin < 2, lammin = min(lam); end
if nargin < 3, lammax = max(lam); end

%% convert
lamt = pi/2*(lam - (lammin + lammax)/2)/(lammax - lammin);

end
