function lam = lamt2lam(lamt,lammin,lammax)
%lamt2lam  Convert numbers back from [-pi/4,pi/4]
%   Y = lamt2lam(X,min,max) converts the matrix X back from the interval
%   [-pi/4,pi/4] where min was mapped to -pi/4 and max to pi/4.
%
%   See also lam2lamt.

%% convert
lam = lamt*2/pi*(lammax - lammin) + (lammin + lammax)/2;

end
