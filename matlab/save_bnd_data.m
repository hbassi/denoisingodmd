function save_sv_data(t, bnd, tol, E, filename)

%% defaults
if nargin < 5; filename = 'plot_data.mat'; end

%% Save data
save(filename, 't', 'bnd', 'tol', 'E');

end
