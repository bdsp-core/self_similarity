function [idx,traceN] = fcnRemoveOutliers(t,trace,showPlots,fs); 

%% detect and correct outliers / artifacts

% clear all; clc; format compact; 
% load trace972; 
% load trace_2; trace = summa;  %(1:200000); 

%trace = summa;  %(1:200000); 
% showPlots=1; 
% fs = 200; 
% dt = 1/fs; 
% Nt = length(trace); 
% t = (0:Nt-1)*dt; 

[tPeaks,peaks,tTroughs,troughs] = fcnGetPeaksTroughs2(t,trace,5,fs); 

t1 = tPeaks; t2 = tTroughs;
y1 = peaks; y2 = troughs;
tf1 = isoutlier(y1,'ThresholdFactor',5); ind1 = find(tf1); 
tf2 = isoutlier(y2,'ThresholdFactor',5); ind2 = find(tf2); 

yy1 = filloutliers(y1,'previous','ThresholdFactor',5);
yy2 = filloutliers(y2,'previous','ThresholdFactor',5);

% get envelope by interpolation
up = interp1(t1,yy1,t); 
lo = interp1(t2,yy2,t); 
lo = min(lo,up); 
up = max(lo,up); 
m = (up+lo)/2; 

idx = []; % keep track of points that have been "corrected"
ind = find(trace>up); trace(ind) = up(ind); idx = [idx ind]; 
ind = find(trace<lo); trace(ind) = lo(ind); idx = [idx ind];
idx = sort(idx); 

traceN = trace-m; 
d = up - lo; 
traceN = 600*traceN/nanmedian(d); 

ind = find(isnan(traceN)); 
traceN(ind) = 500; 

if showPlots==1; 
    figure(10); clf;
    subplot(511); 
    plot(t,trace,tPeaks,peaks,'r.',tTroughs,troughs,'r.'); 
    title('Signal with peaks and troughs marked')
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 

    subplot(512);

    plot(t1,y1,'r.',t2,y2,'r.',t1(ind1),y1(ind1),'b.',t2(ind2),y2(ind2),'b.'); 
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 

    %% replace outliers
    subplot(513); 
    plot(t1,yy1,'r.',t2,yy2,'r.',t1(ind1),yy1(ind1),'b.',t2(ind2),yy2(ind2),'b.'); 
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 

    %% create envelope by interpolation, use to "smooth" outliers
    plot(t,trace,t,up,t,lo); 
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 

    subplot(514); 
    plot(t,trace); 
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 

    %% remove trend, rescale
    hold on; 
    plot(t,m,'r'); 


    subplot(515); 
    plot(t,traceN); 
    ylim([-1500 1500]); 
    xlim([0 0.1e4]); 
end





