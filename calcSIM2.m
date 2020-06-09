function [zz]=calcSIM2(trace,apnea_period)
fs=200;
    dt = 1/fs; 
    Nt = length(trace); 
    t = (0:Nt-1)*dt; 
% fix outliers
[idx,trace] = fcnRemoveOutliers(t,trace,0,fs); 
artifacts = zeros(size(t)); 
artifacts(idx)=1; 

% find peaks and troughs of respiration signal
r = round((1500/200)*fs);
[up,lo] = envelope(trace,r,'peak');
ind = find(up<lo); 
up(ind)=0; lo(ind)=0; 

% find periods with apnea
d = (up-lo); 
p = prctile(d,95);
th = p/10; 
ind = find(d<th); 
dd = nan(size(d)); dd(ind)=mean(d(ind))-500; 

% connect apneas close to other apneas
apneas = fcnConnectApneas(th,d,fs); 

% find peaks and troughs of envelope
env = (up-lo); 
env = env-prctile(env,5); 
r = round(fs*8000/200); 
[~,lo2] = envelope(env,r,'peak'); % upper envelope of envelope 
env2 = env-lo2; % subtract baseline
[up2,~] = envelope(env2,r,'peak'); % lower envelope of envelope
env2 = env2./up2; % normalize skyline
r = round(fs*5000/200); 
% [tePeaks,epeaks,teTroughs,etroughs,i0,i1] = fcnGetPeaksTroughs3(t,env2,.5,r); 
d = abs(up-lo); 
[tePeaks,epeaks,teTroughs,etroughs,i0,i1] = fcnGetPeaksTroughs4(t,d,800,r); 

z = nan(size(d)); 
z(i0) = up(i0); 
z(i1) = up(i1); 
% get similarities
% get time of peak beginning and end
peakTimes = [];
ct=0; 
for i = 2:length(tePeaks)-1; 
    ct=ct+1; 
    t0 = mean(tePeaks((i-1):i)); 
    t1 = mean(tePeaks(i:(i+1))); 
    peakTimes(ct,:) = [t0 t1]; 
end

% do convolutions
sss = []; 
for i = 1:size(peakTimes,1); 
    
    % get first wave
    ind1 = find(t>=peakTimes(i,1) & t<=peakTimes(i,2));
    s1 = env2(ind1); 
%     
%     s1 = trace(ind1); 
    s1 = (s1-mean(s1))/std(s1); 
    
    % get wave ahead
    i0 = max(1,ind1-length(ind1)); i1 = ind1(1); 
    ind0 = i0:i1; 
    s0 = env2(ind0); 
%     s0 = trace(ind0); 
    s0 = (s0-mean(s0))/std(s0); 

    % get wave behind
    i0 = ind1(end); i1 = min(ind1(end)+length(ind1),Nt);
    ind2 = i0:i1; 
    s2 = env2(ind2); 
%     s2 = trace(ind2); 
    s2 = (s2-mean(s2))/std(s2); 

    % convolution
    temp1 = conv(s1,s0,'same'); 
    temp2 = conv(s1,s2,'same'); 
    temp = (temp1+temp2)/2/length(s1); 
    
    tc(i) = mean(peakTimes(i,1:2)); 
    c(i) = max(temp); 
    
    sss{i} = s0; 
end

% fill in similarity for all times
cc = zeros(size(t)); 
for i = 1:length(tc); 
    if i==1; 
        ind = find(t<tc(i));
    else
        ind = find(t>=tc(i-1) & t<tc(i)); 
    end
    cc(ind) = c(i);  
end
%  plot similarity vs time
zz = nan(size(t)); 
clear ind;ind = find(cc>00.8 & ~artifacts & ~isnan(apnea_period) ); 
zz(ind) = -1500; 