function phi = generate_phi(overlap,N)

%% starting vector
phi = zeros(N,1);
phi(1) = sqrt(overlap);
phi(2:end) = sqrt((1 - phi(1)^2)/(N - 1));

%phi(2) = sqrt(overlap);
%phi(1) = 0;
%phi(3:end) = sqrt((1 - phi(2)^2)/(N - 1));
end
