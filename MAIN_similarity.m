function [centrals, central_hypo, similarity]= MAIN_similarity(hdr, s, stage)
%input is 'hdr' (labels of the signal channels) and 's' (signal itself) and 'stages' with the sleep stages. 
%ouput is a variable of 0's and 1's where 1 indicates an event for central
%or hypopnea variable, and the full night similarity index
%(sim>0.8/totalsleeptime)
%plot shows 15 minute tracings per line with cyan the hypopnea and red the
%central apneas

%find indices of chest and abd
     signal= struct2table(hdr);
     signallabel = signal.signal_labels;
        for i = 1:size(signal,1)
            if strcmp(cell2mat(signallabel(i)) , 'CHEST' )
            chest = i;
            elseif strcmp(cell2mat(signallabel(i)),'ABD')
            abd=i;
            end
        end
trace_abd=s(abd,:);
trace_ch=s(chest,:);
trace=trace_abd+trace_ch;

%calculate total sleep time
TST =(length(stage)-length(find(stage==5))-sum(isnan(stage)))/200/60/60; 

% remove outliers from trace
fs=200;
dt = 1/fs; 
Nt = length(trace_abd); 
t = (0:Nt-1)*dt; 
clear ind idx            
[idx_abd,trace_abd] = fcnRemoveOutliers(t,trace_abd,0,fs); 
[idx_ch,trace_ch] = fcnRemoveOutliers(t,trace_ch,0,fs); 

%find apneas in abdominal band
[up,lo]=fcnCreateEnv(trace_abd);
d=up-lo;

centr_det_abd=nan(size(d));
peakenv=nan(size(d));
wdw=3*60*200; %window of 3 minutes to find value for upper 90% of the difference between upper and lower envelope 
for i=wdw:wdw:length(d)-wdw
   peakenv(i:i+wdw)=prctile(d(i-wdw+1:i),90);
end
clear ind;ind=find(d<0.2*peakenv ); %central apnea when difference between upper and lower envelope is smaller than 0.2 times the peak (upper 90%) of the 3 minutes before
centr_det_abd(ind)=-1200;
    
centr_hypo_abd=nan(size(d));
clear ind;ind=find(d<0.7*peakenv ); %central hypopnea when difference between upper and lower envelope is smaller than 0.7 times the peak (upper 90%) of the 3 minutes before
centr_hypo_abd(ind)=-1500;

%find apneas in chest band
[up,lo]=fcnCreateEnv(trace_ch);
d=up-lo;

centr_det_ch=nan(size(d));
peakenv=nan(size(d));
for i=wdw:wdw:length(d)-wdw
   peakenv(i:i+wdw)=prctile(d(i-wdw+1:i),90);
end
clear ind;ind=find(d<0.2*peakenv );
centr_det_ch(ind)=-2000;

centr_hypo_ch=nan(size(d));
clear ind;ind=find(d<0.7*peakenv);
centr_hypo_ch(ind)=-1500;

%combine chest and abdomen
centr_det=centr_det_abd+centr_det_ch;
clear ind;ind=find(centr_det==-3200);
centr_detected=nan(size(trace_abd));
centr_detected(ind)=-500;

centr_hypo=centr_hypo_abd+centr_hypo_ch;
clear ind;ind=find(centr_hypo==-3000);
centr_hypo_=nan(size(trace_abd));
centr_hypo_(ind)=-700;

%it is only a central apnea when it lasts 10 sec, and during nonwake stage
%9 seconds is used now, because the envelope causes a more smooth decrease
%of the trace so have to give it a bit more margin
centr_detected_final=nan(size(centr_detected));
for i=1:length(centr_detected)-(9*200)
    if ~isnan(centr_detected(i)) && stage(i)~=5 && ~isnan(stage(i))  
        if ~isnan(centr_detected(i:i+(9*200))) 
            centr_detected_final(i:i+(9*200))=-800;
        end
    end
end

%get 10 sec hypopneas
centr_hypo_final=nan(size(centr_hypo));
for i=1:length(centr_hypo_)-(9*200)
    if ~isnan(centr_hypo_(i)) && stage(i)~=5 && ~isnan(stage(i))  
        if ~isnan(centr_hypo_(i:i+(9*200))) 
            centr_hypo_final(i:i+(9*200))=-800;
        end
    end
end

% periods of hypo/apnea region, 2 minute before + after
apnea_region=nan(size(trace_abd));
clear ind;ind=find(~isnan(centr_detected_final));

for i=1:length(ind)
apnea_region(ind(i)-2*60*200:ind(i)+2*60*200)=-1000;
end

clear ind;ind=find(~isnan(centr_hypo_final));
for i=1:length(ind)
apnea_region(ind(i)-2*60*200:ind(i)+2*60*200)=-1000;
end

%calculate similarity>0.8 during hypo/apnea period
apnea_region=apnea_region(1:length(trace_abd));
[zz]=calcSIM2(trace,apnea_region);

%duration similarity divided by total sleep time (in samples)
lengthsim=find(~isnan(zz));
similarity=length(lengthsim)/TST;

%true central hypopneas are hypopneas with high similarity
centr_hypo_final2=nan(size(centr_hypo_final));
clear ind; ind=find(~isnan(centr_hypo_final) & ~isnan(zz));
centr_hypo_final2(ind)=-2200;

%fill up gaps caused by fluctuating similarity around 0.8
hypos=zeros(size(trace_abd));
clear ind;ind=find(centr_hypo_final2==-2200);
hypos(ind)=1;

shift=[0 hypos(1:end-1)];
StateSwitches = hypos - shift;
HypOn=find(StateSwitches==1);
HypOff=find(StateSwitches==-1);
for i=1:length(HypOn)-1
    if HypOn(i+1)-HypOff(i)<3*200 %3 sec: if gap between 2 hypos is smaller than 3 seconds, the gap is considered central hypoventilation as well
        centr_hypo_final2(HypOff(i):HypOn(i+1))=-2200;
    end
end

%get only the 10 sec duration central hypopneas again (but again 9 seconds)
centr_hypo_final3=nan(size(centr_hypo_final2));
for i=1:length(centr_hypo_final2)-(9*200)
    if ~isnan(centr_hypo_final2(i)) && stage(i)~=5 && ~isnan(stage(i))  
        if ~isnan(centr_hypo_final2(i:i+(9*200))) 
            centr_hypo_final3(i:i+(9*200))=-800;
        end
    end
end
centr_hypo_final2=centr_hypo_final3;

%get variable of central apneas with only 0's and 1's
centrals=zeros(size(trace_abd));
clear ind;ind=find(centr_detected_final==-800);
centrals(ind)=1;

%get variable of central hypopneas with only 0's and 1's
central_hypo=zeros(size(trace_abd));
clear ind;ind=find(centr_hypo_final2==-800);
central_hypo(ind)=1;

%plot
figure;fcnDrawBreaths(trace_abd,'k',fs);hold on;
fcnShadeSimilarities(centr_hypo_final2,'c',fs); 
 fcnShadeSimilarities(centr_detected_final,'r',fs);  
end
