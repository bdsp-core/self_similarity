function fcnDrawBreaths(trace,col,fs);

% trace = summa;  %(1:200000); 
dt = 1/fs; 
Nt = length(trace); 
t = (0:Nt-1)*dt; 

%% 
n = 15*60*fs; % number of samples per row
nRow = ceil(Nt/n);
i0 = 1; 
xx = [];
yy = nan(nRow,n);
for i = 1:nRow; 
    i1 = min(i0+n-1,Nt); 
    ind = i0:i1; 
    xx = t(1:n); 
    yy(i,1:length(ind)) = -i*4000+trace(ind); 
    i0 = i1+1;
end

plot(xx,yy,'color',col,'LineWidth',0.5); 
axis off
set(gcf,'color','w'); 

ax1 = gca;                   % gca = get current axis
ax1.YAxis.Visible = 'off';