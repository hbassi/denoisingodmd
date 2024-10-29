function H = make_hankel(data,m,n)

%% defaults
if nargin < 3, n = m; end

%% check
assert(length(data) >= m+n-1);

%% Hankel matrix
H = hankel(data(1:m),data(m:m+n-1));

end
