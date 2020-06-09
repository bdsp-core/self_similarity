function [up,lo]=fcnCreateEnv(trace)

fs=200;
r = round((1000/200)*fs);
[up,lo] = envelope(trace,r,'peak');
ind = find(up<lo); 
up(ind)=0; lo(ind)=0; 
clear ind
ind = find(up<(0.8*trace));

for i=801:length(ind)
up(ind(i))=mean(up(ind(i)-600:ind(i)));
end

clear ind
ind = find(lo>0.9*trace);
for i=800:length(ind)
lo(ind(i))=mean(lo(ind(i)-600:ind(i)));
end
end


