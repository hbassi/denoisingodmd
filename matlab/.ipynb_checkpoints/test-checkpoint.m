function test(varargin)
% Parse command line arguments
p = inputParser;
addParameter(p, 'molecule', 'Cr2', @ischar);
addParameter(p, 'noise', 1.0, @isnumeric);
addParameter(p, 'overlap', 0.2, @isnumeric);
addParameter(p, 'tol', [0.01, 0.1, 0.9999], @isnumeric);
addParameter(p, 'Tmax', 500, @isnumeric);
addParameter(p, 'dt', 1, @isnumeric);

parse(p, varargin{:});

molecule = p.Results.molecule;
noise = p.Results.noise;
overlap = p.Results.overlap;
tol = p.Results.tol;
Tmax = p.Results.Tmax;
dt = p.Results.dt;

fun = @dmd;
step = 1;
datatype = 'r';   % r: real, i: imaginary, c: complex
noisetype = 'r';  % r: real, i: imaginary, c: complex
savedata = 1;

%% fix random seed
rng(100);

%% Cr2 data (true eigenvalues + HF ground state)
if strcmp(molecule,'Cr2')
    load('Cr2_4000.mat');
elseif strcmp(molecule,'LiH')
    load('LiH_2989.mat');
elseif strcmp(molecule,'HeH+')
    load('heh+_sto-3g_hamiltonian.mat');
    E = E';
elseif strcmp(molecule,'H6')
    load('H6_200.mat');
else
    error('Wrong molecule!');
end
Et = lam2lamt(E,E(1) - 0.2, E(end) + 0.2);  % convert to [-pi/4,pi/4]
%Et = E/100;


if overlap == -1
    [dataH,dataS] = generate_samples(Et,psiHF,dt,Tmax);
else
    phi = generate_phi(overlap,length(E));
    [dataH,dataS] = generate_samples(Et,phi,dt,Tmax);
end

%% add noise
if strcmpi(noisetype,'r') || strcmpi(noisetype,'c')
    dataH = dataH + noise*randn(Tmax,1);
    dataS = dataS + noise*[0;randn(Tmax-1,1)];
elseif strcmpi(noisetype,'i') || strcmpi(noisetype,'c')
  dataH = dataH + 1i*noise*[0;randn(Tmax-1,1)];
    dataS = dataS + 1i*noise*[0;randn(Tmax-1,1)];
end
if strcmpi(datatype,'r')
    dataH = real(dataH);
    dataS = real(dataS);
elseif strcmpi(datatype,'i')
   dataH = imag(dataH);
   dataS = imag(dataS);
end

data = load('./noisy_data/denoised_dataS_Cr2_noise=0.1_Tmax=1000_overlap=0.2_dt=1.mat');
dataS = data.dataS.';

%% run
% Use the empty list for denoising (denoised trajectories are loaded in during run_compare), and dataS for vanilla
[lamt,t,svs,bnds] = run_compare(dataH,dataS,dt,fun,tol,Tmax,step);
lam = lamt2lam(lamt,E(1) - 0.2,E(end) +0.2);  % convert back from [-pi/4,pi/4]
%lam = lamt * 100;
%% save
if savedata
    if overlap == -1
        overlap_str = 'HF';
    else
        overlap_str = ['o', num2str(100*overlap)];
    end
    tol_str = strrep(num2str(tol), ' ', '_');
    filename = [molecule, '-n', num2str(noise), '-', overlap_str, '-tol', tol_str, '-Tmax', num2str(Tmax), '-dt', num2str(dt), '-', func2str(fun), '.dat'];
    %writematrix([t(:) abs(lam - E(1))], filename, 'Delimiter', '\t');
end

%% plot
mytitle = [molecule,' (noise = ',num2str(noise)];
if overlap == 0, mytitle = [mytitle,'  -  HF)'];
else,            mytitle = [mytitle,'  -  overlap = ',num2str(overlap),')']; end
if isequal(fun,@dmd),       mytitle = ['DMD Method: ',mytitle];
elseif isequal(fun,@mp),    mytitle = ['Matrix Pencil Method: ',mytitle];
elseif isequal(fun,@vqpe),  mytitle = ['VQPE: ',mytitle];
elseif isequal(fun,@uvqpe), mytitle = ['Unitary VQPE: ',mytitle];
end
save_plot_data(t, lam, tol, E, [molecule, '-n', num2str(noise), '-', overlap_str, '-tol', tol_str, '-Tmax', num2str(Tmax), '-dt', num2str(dt), '-_fourier_denoised_complex_ff_left_right.mat']);
%save_plot_data(t, lam, tol, E, [molecule, '-n', num2str(noise), '-', overlap_str, '-tol', tol_str, '-Tmax', num2str(Tmax), '-dt', num2str(dt), '-_real_plot_data.mat']);
%plot_compare(t, lam, tol, E, mytitle, [0, Tmax], [1e-6, 1]);
