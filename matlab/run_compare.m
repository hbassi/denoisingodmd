function [lam, t, svs, bnds] = run_compare(dataH, dataS, dt, fun, tol, Tmax, step)

%% defaults
if nargin < 4; fun = @dmd; end
if nargin < 5; tol = [1e-1, 1e-2, 1e-3]; end
if nargin < 6; Tmax = 500; end
if nargin < 7; step = 10; end

%% eigenvalue approximation
t = 15:step:Tmax;
lam = inf(length(t), length(tol), 2); 
svs = inf(length(t), length(tol));
bnds = inf(length(t), length(tol));

for i = 1:length(t)
    for j = 1:length(tol)
        % Comment out 18 - 26 for vanilla
        %Tmax_str = num2str(t(i)); 
        %file_name = ['./noisy_data/noisy_dataS_Cr2_noise=0.1_Tmax=', Tmax_str, '_overlap=0.2_dt=1.mat'];
        %file_name = ['./denoised_data/denoised_dataS_Cr2_GP_Tmax=', Tmax_str, '_overlap=0.2noise=0.001_real_multi_traj_std_initial_guess.mat'];
        
        % Load the file corresponding to this Tmax
        %data = load(file_name); 
        %dataS = data.denoised_dataS;
        %dataS = data.dataS;
        % Data is stored as transpose from py script
        %dataS = dataS.';
        if isequal(fun, @vqpe)
            [omega, sv] = fun(dataH(1:t(i)), dataS(1:t(i)), tol(j));
            lam(i, j, 1) = min(omega); 
            lam(i, j, 2) = NaN;       
        else
            [omega, sv, bnd] = fun(dataS(1:t(i)), dt, tol(j));
            imag_vals = imag(omega); 
            sorted_imag_vals = sort(imag_vals, 'descend');
            
            if length(sorted_imag_vals) >= 1
                lam(i, j, 1) = -sorted_imag_vals(1); % First most negative
            end
            if length(sorted_imag_vals) >= 2
                lam(i, j, 2) = -sorted_imag_vals(2); % Second most negative
            else
                lam(i, j, 2) = NaN; % If there is no second value, set to NaN
            end
            
            svs(i, j) = sv;
            bnds(i, j) = bnd;
        end
    end
end
