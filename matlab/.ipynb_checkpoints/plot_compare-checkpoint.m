function plot_compare(t,lam,tol,E,mytitle,xlimits,ylimits)

%% defaults
if nargin < 5; mytitle = ''; end
if nargin < 6; xlimits = []; end
if nargin < 7; ylimits = []; end

%% plot
figure;
marker = '*ods';
lgnd = cell(size(tol));
for i = 1:length(tol)
    mark = marker(mod(i-1,length(marker))+1);
    semilogy(t,abs(lam(:,i) - E(1)),mark); hold on
   %semilogy(t,abs(lam(:,i) - E(1))/abs(E(1)),mark); hold on
    lgnd{i} = ['tol = ',num2str(tol(i))];
end
plot([0,t(end)],[1e-3,1e-3],':k');
legend(lgnd);
xlabel('# timesteps');
ylabel('absolute error');
%label('relative error');
if ~isempty(mytitle); title(mytitle); end
if ~isempty(xlimits); xlim(xlimits); end
if ~isempty(ylimits); ylim(ylimits); end

end
