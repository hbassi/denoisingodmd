function save_plot_data(t, lam, tol, E, filename)

%% defaults
if nargin < 5; filename = 'plot_data.mat'; end

%% Save data
save(filename, 't', 'lam', 'tol', 'E');

end
