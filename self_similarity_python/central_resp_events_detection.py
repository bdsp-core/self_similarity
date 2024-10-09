
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import groupby
from scipy.signal import find_peaks, detrend
from statsmodels.robust.scale import mad
from scipy import interpolate
import warnings
warnings.filterwarnings("ignore")
import glob
import mne
from mne.filter import filter_data, notch_filter, resample
from mne.time_frequency import psd_array_multitaper
from tqdm import tqdm 
import datetime
from datetime import timedelta

def isNaN(num):
    return num != num

def envelope(trace, fs=200, r=1000):
    '''
    fs: sampling rate
    r: min distance of peaks, in samples
    '''
    peaks = find_peaks(trace, distance=r)[0]
    troughs = find_peaks(-trace, distance=r)[0]
    
    mvg_avg_dur = 10 # in seconds
    trace_mvg_avg = pd.Series(trace)
    trace_mvg_avg = trace_mvg_avg.rolling(mvg_avg_dur*fs, center=True).mean().values
    trace_mvg_avg[np.isnan(trace_mvg_avg)]=0
    
    peaks = peaks[np.where(trace[peaks] > trace_mvg_avg[peaks]*1.1)[0]]
    troughs =  troughs[np.where(trace[troughs] < trace_mvg_avg[troughs]*0.9)[0]]

    if peaks.shape[0] == 0:
        return None, None

    try:
        f_interp = interpolate.interp1d(peaks, trace[peaks], kind='cubic', fill_value="extrapolate")
        envelope_up = f_interp(range(len(trace)))
    except:
        envelope_up = None
    try:
        f_interp = interpolate.interp1d(troughs, trace[troughs], kind='cubic', fill_value="extrapolate")
        envelope_lo = f_interp(range(len(trace)))
    except:
        envelope_lo = None

    return envelope_up, envelope_lo


def create_env(trace, fs, r = 5):
    'r: in seconds'
    
    r = r*fs; # 5 seconds distance for peaks.
    up,lo =  envelope(trace, fs, r=r)

    if np.any([up is None, lo is None]):
        return up, lo

    ind = np.where(up<lo)[0]
    up[ind] = 0
    lo[ind] = 0
    ind = np.where(up<(0.8*trace))[0]
    for i in range(4*fs,len(ind)):
        up[ind[i]] = np.mean(up[(ind[i]-3*fs):ind[i]])
    ind = np.where(lo>(0.9*trace))[0]
    for i in range(4*fs,len(ind)):
        lo[ind[i]] = np.mean(lo[(ind[i]-3*fs):ind[i]])
    return up,lo

def connect_apneas(th,d,fs):
    Nt = len(d)
    n = round(fs*5000/200); # max time before giving up search for successor

    # forward
    ct = float("inf")
    inOne = 0
    z = np.full([ len(d)], np.nan)
    for i in range(Nt):
        if d[i]<th : 
            ct=0
        if d[i]>=th:
            ct=ct+1
        if ct<n:
            z[i] = -1000

    #backward
    dd = np.fliplr([d])[0]
    ct = float("inf")
    inOne = 0
    zz=np.full([ len(dd)], np.nan)
    for i in range(Nt):
        if dd[i]<th:
            ct=0;
        if dd[i]>=th:
            ct=ct+1; 
        if ct<n:
            zz[i] = -1000  
    zz = np.fliplr([zz])[0]
    ind = np.where(~isNaN(zz))
    z[ind] = -1000;
    return z


def remove_outliers(t,trace,fs, min_height=0.01, do_plots=False):
    
    tPeaks,peaks,tTroughs,troughs = get_peaks_troughs_2(t,trace,min_height,fs)
    t1 = tPeaks.copy()
    t2 = tTroughs.copy()
    y1 = peaks.copy()
    y2 = troughs.copy()
    ThresholdFactor = 5
    tf1 = is_outlier(y1,ThresholdFactor=ThresholdFactor)
    ind1 = list(np.nonzero(tf1)[0]); 
    tf2 = is_outlier(y2,ThresholdFactor=ThresholdFactor)
    ind2 = list(np.nonzero(tf2)[0]); 

    yy1 = y1.copy()
    yy2 = y2.copy()
    yy1 = fill_outliers(yy1,ThresholdFactor=ThresholdFactor)
    yy2 = fill_outliers(yy2,ThresholdFactor=ThresholdFactor)

    #get envelope by interpolation
    f1 = interpolate.interp1d(t1,yy1,bounds_error=False)
    f2 = interpolate.interp1d(t2,yy2,bounds_error=False)
    up = f1(t)
    lo = f2(t)
    lo = [ lo[i] if (lo[i] <= up[i]) else up[i] for i in range(len(lo))]
    up = [ lo[i] if (lo[i] >= up[i]) else up[i] for i in range(len(up))]
    m = np.add(up,lo)/2

    idx = [] # keep track of points that have been "corrected"
    ind = [ i for i in range(len(trace)) if (trace[i] > up[i]) ]
    idx = idx + ind
    trace = [ up[i] if (trace[i] > up[i]) else trace[i] for i in range(len(trace))]
    ind = [ i for i in range(len(trace)) if (trace[i] < lo[i]) ]
    idx = idx + ind
    trace = [ lo[i] if (trace[i] < lo[i]) else trace[i] for i in range(len(trace))]
    idx = np.sort(idx)

    traceN = trace-m; 
#     d = np.array(up) - np.array(lo); 
#     print(np.nanmedian(d))
#     traceN = 600*traceN/np.nanmedian(d); 

    traceN[np.isnan(traceN)] = 0 # 500
    

    if do_plots:
        fig, ax = plt.subplots(5,1, sharex=True, sharey=True, figsize=(16,20), dpi=80)
        ax[0].plot(t, trace,linewidth=1)
        ax[0].scatter(tPeaks, peaks, c='r',s=8)
        ax[0].scatter(tTroughs, troughs, c='r',s=8)
        ax[0].set_xlim([0,1000])
        ax[0].set_ylim([-11,11])
        ax[1].scatter(t1,y1,c='r',s=8)
        ax[1].scatter(t2, y2, c='r',s=8)
        ax[1].scatter(t1[ind1],y1[ind1],c='blue',s=8)
        ax[1].scatter(t2[ind2],y2[ind2],c='blue',s=8)    

    #     ax[2].scatter(t1,yy1,c='r',s=8)
    #     ax[2].scatter(t2,yy2,c='r',s=8)
    #     ax[2].scatter(t1[ind1],yy1[ind1],c='blue',s=8)
    #     ax[2].scatter(t2[ind2],yy2[ind2],c='blue',s=8)
        ax[2].plot(t,trace,linewidth=1)
        ax[2].plot(t,up,c='peru',linewidth=1)
        ax[2].plot(t,lo,c='gold',linewidth=1)

        ax[3].plot(t,trace,linewidth=1)
        ax[3].plot(t,m,c='r',linewidth=1)

        ax[4].plot(t, traceN, linewidth=1)
    
    return idx,traceN

# replaces outliers with the previous non-outlier element
def fill_outliers(y,ThresholdFactor=5):
    outliers = is_outlier(y,ThresholdFactor=ThresholdFactor)
    outliers_indices = np.nonzero(outliers);
    for outlier_index in list(outliers_indices[0]):
        i=1
        while (outliers[outlier_index-i] == 1):
            i += 1
        y[outlier_index] = y[outlier_index-i] 
    return y

def is_outlier(x_array,ThresholdFactor=5):
    MAD = mad(x_array)
    median = np.median(x_array)
    return [1 if ((x > (median+MAD*ThresholdFactor))|(x < (median-MAD*ThresholdFactor))) else 0 for x in x_array] 




def get_peaks_troughs_4(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i1 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind])
        jj = list(trace[ind]).index(ii) 
        i1.append(ind[jj])

    if len(i0) == 0:
        return None, None, None, None, None, None

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind])
    jj = list(trace[ind]).index(ii) 
    i1.append(ind[jj])
    i1=np.asarray(i1)

    tPeaks =  t[i0] 
    peaks = trace[i0]; 
    tTroughs = t[i1]; 
    troughs = trace[i1]; 

    killIt = np.zeros(len(tPeaks))
    for i in range(0,len(tPeaks)):
        if (peaks[i]/(troughs[i]+.1)<3.5): #3  Eline: 5
            killIt[i]=1
        if troughs[i]>0.9: #1.2 Eline: 100 (based on absolute amplitude)
            killIt[i]=1
        if (abs(tPeaks[i]-tTroughs[i])>100):
            killIt[i]=1

    ind = list(np.where(killIt == 0)[0])
    peaks = peaks[ind] 
    troughs= troughs[ind]
    tPeaks=tPeaks[ind]
    tTroughs=tTroughs[ind]
    i0 = i0[ind]
    i1 = i1[ind]
    return tPeaks,peaks,tTroughs,troughs,i0,i1

def get_peaks_troughs_2(t,trace,mph,mpd):
    i0,properties = find_peaks(trace,height=mph,distance=mpd)
    i2 = [];
    for i in range(0,len(i0)-1):
        ind = list(np.arange(i0[i],i0[i+1]))
        ii = min(trace[ind]); 
        jj = list(trace[ind]).index(ii) 
        i2.append(ind[jj]); 

    ind = list(np.arange(i0[-1],len(trace)))
    ii = min(trace[ind]); 
    jj = list(trace[ind]).index(ii) 
    i2.append(ind[jj]);

    tPeaks =  t[i0].copy()
    peaks = trace[i0].copy(); 
    tTroughs = t[i2].copy(); 
    troughs = trace[i2].copy(); 
    return tPeaks,peaks,tTroughs,troughs


def load_mgh_matlab_data(data_path, fs):
    
    data = loadmat(data_path)
    channels = [data['hdr'][0][x][0][0] for x in range(len(data['hdr'][0]))]
    abd_channel_no = channels.index('ABD')
    chest_channel_no = channels.index('CHEST')
    trace = data['s'][abd_channel_no,:] + data['s'][chest_channel_no,:]
    t = np.arange(len(trace))/fs
    
    return t, trace


def clip_z_normalize(signal):
    signal_clipped = np.clip(signal, np.percentile(signal,2), np.percentile(signal,98))
    signal = (signal - np.mean(signal_clipped))/np.std(signal_clipped)
    return signal


def detect_central_events(lo,up,fs):

    d = up - lo 
    centr_detected = np.empty(d.shape)
    peakenv= np.empty(d.shape)
    wdw=3*60*fs #window of 3 minutes to find value for upper 90% of the difference between upper and lower envelope 
    for i in range(wdw, len(d)-wdw, wdw):
        peakenv[i:i+wdw]=np.percentile(d[i-wdw+1:i],90)

    # central apnea when difference between upper and lower envelope is smaller than 0.2 times the 90% percentile peakenv (difference of upper and lower envelope) of the 3 minutes before
    centr_detected[np.where(d<0.2*peakenv)[0]]=1;

    # HYPOPNEAS
    centr_hypo=np.empty(d.shape)
    centr_hypo[np.where(d<0.7*peakenv)[0]]=1;
    
    # %it is only a central apnea when it lasts 10 sec
    # %9 seconds is used now, because the envelope causes a more smooth decrease
    # %of the trace so have to give it a bit more margin
    centr_detected_final=np.zeros(centr_detected.shape);
    for i in np.arange(4.5*fs, len(centr_detected)-4.5*fs, fs):
        if sum(centr_detected[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_detected_final[int(i-4.5*fs):int(i+4.5*fs)] = 1


    centr_hypo_final=np.zeros(centr_hypo.shape);
    for i in np.arange(4.5*fs, len(centr_hypo)-4.5*fs, fs):
        if sum(centr_hypo[int(i-4.5*fs):int(i+4.5*fs)])>8.5*fs:
            centr_hypo_final[int(i-4.5*fs):int(i+4.5*fs)] = 1
          

    return centr_detected_final, centr_hypo_final

def generate_report(trace, centr_detected_final, centr_hypo_final, fs, save=False):
    
    state_switches = centr_detected_final[1:] - centr_detected_final[:-1]
    apnea_start = np.where(state_switches==1)[0]+1
    apnea_end = np.where(state_switches==-1)[0]+1
    state_switches = centr_hypo_final[1:] - centr_hypo_final[:-1]
    hypopnea_start = np.where(state_switches==1)[0]+1
    hypopnea_end = np.where(state_switches==-1)[0]+1
    
    hypopnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    hypopnea_report.start_sample = hypopnea_start
    hypopnea_report.end_sample = hypopnea_end
    hypopnea_report.start_second = np.round(hypopnea_start/fs,1)
    hypopnea_report.end_second = np.round(hypopnea_end/fs,1)
    hypopnea_report.event  = ['central hypopnea']*hypopnea_report.shape[0]
    
    apnea_report = pd.DataFrame(columns=['start_sample','end_sample','start_second','end_second','event'])
    apnea_report.start_sample = apnea_start
    apnea_report.end_sample = apnea_end
    apnea_report.start_second = np.round(apnea_start/fs,1)
    apnea_report.end_second = np.round(apnea_end/fs,1)
    apnea_report.event  = ['central apnea']*apnea_report.shape[0]
    
    report_events = pd.concat([hypopnea_report,apnea_report],ignore_index=True).sort_values(by='start_sample').reset_index()
    report_events.drop('index', axis=1,inplace=True)
    
    summary_report = pd.DataFrame([],columns=['signal duration (h)', 'detected central apnea events', 'detected central hypopnea events'])
    summary_report['signal duration (h)'] = [np.round(len(trace)/fs/3600,2)]
    summary_report['detected central apnea events'] = [len(apnea_start)]
    summary_report['detected central hypopnea events'] = [len(hypopnea_start)]
    
    full_report = pd.concat([report_events,summary_report],axis=1)
    if save:
        full_report.to_csv('report_central_resp_events.csv',index=False)

    return full_report
    

# replace all zeros by nan's

def plot_central_events(trace, centr_detected_final, centr_hypo_final, savepath = 'figure_central_resp_events'):

    central_events = np.zeros(centr_detected_final.shape)
    central_events[centr_hypo_final.astype('bool')] = 2
    central_events[centr_detected_final.astype('bool')] = 1
    central_events.astype('float')
    central_events[central_events==0] = float('nan')

    assert(central_events.shape[0] == trace.shape[0])


    # use seg_start_pos to convert to the nonoverlapping signal
    # y = ytrue                               # shape = (N, 4100)
    # yp = apnea_prediction                   # shape = (N, 4100)
    # yp_smooth = apnea_prediction_smooth     # shape = (N, 4100)

    # define the ids each row
    nrow = 10
    row_ids = np.array_split(np.arange(len(trace)), nrow)
    row_ids.reverse()

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    row_height = 10
    label_color = [None, 'g', 'c']

    # here, we get clip-normalized signal. we should not need to plot >3 STD values:
    trace[np.abs(trace > 3)] = np.nan

    for ri in range(nrow):
        # plot signal
        ax.plot(trace[row_ids[ri]]+ri*row_height, c='k', lw=0.2)


        y2 = central_events         

        yi=0

        # run over each plot row
        for ri in range(nrow):
            # plot tech annonation
    #         ax.axhline(ri*row_height-3*(2**yi), c=[0.5,0.5,0.5], ls='--', lw=0.2)  # gridline
            loc = 0

            # group all labels and plot them
            for i, j in groupby(y2[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    # i is the value of the label
                    # list(j) is the list of labels with same value
                    ax.plot([loc, loc+len_j], [ri*row_height-3*(2**yi)]*2, c=label_color[int(i)], lw=1)
                loc += len_j

    # plot layout setup
    ax.set_xlim([0, max([len(x) for x in row_ids])])
    ax.axis('off')
    plt.tight_layout()
    # plt.title(test_info[si])
    # save the figure
    plt.savefig(savepath + '.pdf')
    plt.savefig(savepath + '.png')


def central_respiration_event_detection(data_path,fs):
    
    # loads abdomen+chest respiration signal from an MGH matlab format file
    t, trace = load_mgh_matlab_data(data_path, fs)

    # normalize respiration signal
    trace = clip_z_normalize(trace)

    # remove outliers from signal based on signal envelope
    idx,trace = remove_outliers(t,trace,fs, min_height=0.01)

    # compute the upper and lower envelope of the respiration signal
    [up,lo]=create_env(trace, fs)

    # based on envelopes, central respiration signals can be detected
    apneas, hypopneas = detect_central_events(lo,up,fs)

    similarity_array = compute_similarity(up, lo, fs)

    apneas, hypopneas = post_processing_detections(apneas, hypopneas)

    # saves report in current folder
    report = generate_report(trace, apneas, hypopneas, fs)
    # saves plot in current folder
    plot_central_events(trace, apneas, hypopneas)

def compute_similarity(up, lo, fs):
    
    env_diff = up-lo
    similarity_array = np.zeros(env_diff.shape)

    env_diff[env_diff<0] = 0
    th = np.percentile(env_diff, 95)
    
    apneas2 = connect_apneas(th, env_diff, fs)
    apneas2[np.where(np.isnan(apneas2))] = 0
    apneas2[np.where(apneas2==-1000)] = 2

    env_diff = env_diff-np.percentile(env_diff,5)
    # upper and lower envelopes of envelope differences
    [tmp,lo2] = create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if lo2 is None:
        return similarity_array

    # subtract baseline
    env_diff = env_diff - lo2 
    [up2,tmp]=create_env(env_diff, fs=fs, r=25) # 25 8000/fs
    if up2 is None:
        return similarity_array

    env_diff = env_diff/(up2+0.00001)
    t = np.arange(len(env_diff))/fs
    
    [tePeaks,epeaks,teTroughs,etroughs,i0,i1] = get_peaks_troughs_4(t,abs(up-lo),2.51,10*fs) # get_peaks_troughs_4(t,abs(up-lo),3.51,5000)

    if tePeaks is None:
        return similarity_array

    # get similarities
    # get time of crescendo-diminuendo patterns

    # as done in Eline's script. peakTimes are start/end points of waves/segments of signals of interest.
    # peakTimes are simply mean points between two peaks found in envelope differnece.
    # for first peak, it's -15/+15 seconds of 
    # new implemention, with samples:
    pattern_boundary = np.zeros((len(i0),2), dtype=np.int64) # this is called peakTimes in Eline's code.
    similarity = []
    if pattern_boundary.shape[0] > 0:
        pattern_boundary[0,:] = [max(0,i0[0]-15*fs), i0[0]+15*fs]
        for i in range(len(tePeaks)-1):
            pattern_boundary[i+1,:] = [np.mean(i0[i:i+2]), np.mean(i0[i+1:i+3])]
        if len(tePeaks)-1 > 0:
            pattern_boundary[i+1,1] = min(pattern_boundary[i+1,1]+15*fs, len(env_diff))    


        # do convolution:

        for [pattern_start, pattern_end] in pattern_boundary:
            if pattern_start == 0: continue
            # get first pattern (wave) and normalize it:
            s1 = env_diff[pattern_start:pattern_end].copy()
            s1 = (s1-np.mean(s1))/np.std(s1+0.00001)
            len_pattern = pattern_end-pattern_start
            # get wave ahead (wave-1)|
            s0 = env_diff[max(1,pattern_start-len_pattern):pattern_start].copy()
            s0 = (s0-np.mean(s0))/np.std(s0+0.00001)

            # get wave behind (wave+1)
            s2 = env_diff[pattern_end: min(pattern_end+len_pattern,len(env_diff))]
            s2 = (s2-np.mean(s2))/np.std(s2+0.00001)
            # convolution
            if len(s2)==0:
        #             s2 = np.zeros(s1.shape)
                average_conv = max(np.convolve(s1,s0,'same'))/len_pattern
            else:
                conv1 = np.convolve(s1,s0,'same')
                conv2 = np.convolve(s1,s2,'same')
        #         average_conv = (conv1 + conv2)/2/len_pattern
                average_conv = (np.percentile(conv1,99)/len_pattern +  np.percentile(conv2,99)/len_pattern)/2

        #     similarity.append(max(average_conv))
            similarity.append(average_conv)
            similarity_array[pattern_start:pattern_end] = average_conv
            #     plt.figure()
            #     plt.plot(s1)
            #     plt.plot(s0)
            #     plt.plot(s2)
            #     plt.legend(['center wave', 'wave ahead', 'wave behind'])
            #     plt.title('current implementation, similarity: ' + str(average_conv))
        similarity = np.array(similarity)

    return similarity_array
    

def post_processing_detections(apneas, hypopneas, similarity_array, fs):

    # if apnea in hypopnea, keep only apnea.
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if any(apneas[hypo_on_tmp:hypo_off_tmp]==1):
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0

    # remove detections with similarity less than 0.5
    apneas[similarity_array<0.5] = 0
    hypopneas[similarity_array<0.5] = 0
    
    # need to last more than 9 seconds:
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    diff=[]
    for [hypo_on_tmp, hypo_off_tmp] in list(zip(hypo_on, hypo_off)):
        if hypo_off_tmp - hypo_on_tmp < 9*fs:
            hypopneas[hypo_on_tmp:hypo_off_tmp+1] = 0
    
    # if less than 5 seconds gap between two detections, connect them
    
    hypo_on = np.where(np.diff(hypopneas)==1)[0]
    hypo_off = np.where(np.diff(hypopneas)==-1)[0]
    apnea_on = np.where(np.diff(apneas)==1)[0]
    apnea_off = np.where(np.diff(apneas)==-1)[0]
        
    for [hypo_on_tmp, hypo_off_previous] in list(zip(hypo_on[1:], hypo_off[:-1])):
        if hypo_on_tmp - hypo_off_previous < 5*fs:
            hypopneas[hypo_off_previous:hypo_on_tmp+1] = 1

    for [apnea_on_tmp, apnea_off_previous] in list(zip(apnea_on[1:], apnea_off[:-1])):
        if apnea_on_tmp - apnea_off_previous < 5*fs:
            apneas[apnea_off_previous:apnea_on_tmp+1] = 1

    return apneas, hypopneas


def main():

    import time
    starttime = time.time()

    self_sim_results_dir = 'central_resp_events'
    edf_folder = '.'

    # get all edf paths in this folder
    edf_paths = glob.glob(os.path.join(edf_folder, '*.edf'))  
      
    notch_freq = 60
    bandpass_freq = [0, 10]
    n_jobs = 1
    new_fs = 10

    # if result folder does not exist, create
    if not os.path.exists(self_sim_results_dir):
        os.mkdir(self_sim_results_dir)

    for edf_path in tqdm(edf_paths):

        try:
        # if 1:
            # load edf
            respiratory_effort_channels = ['Abdomen', 'Abdominal', 'Chest', 'ABDOMEN', 'CHEST', 'ABDOMINAL', 'Effort THO']
            edf = mne.io.read_raw_edf(edf_path, stim_channel=None, preload=False, verbose=False)
            edf_channels = edf.info['ch_names']
            fs = int(edf.info['sfreq'])

            try:
                start_time = datetime.datetime.fromtimestamp(edf.info['meas_date'][0]) + timedelta(seconds=time.altzone)
            except:
                start_time = edf.info['meas_date'] + timedelta(seconds=time.altzone)

            # find respiratory effort channels

            if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
                print(f'{edf_path}:')
                print(f'Expected Channel Name not found. Try flexible search for anything with "effort", "chest", "tho", or "abd" in it.')
                respiratory_effort_channels = [x for x in edf_channels if any([y in x.lower() for y in ['effort', 'chest', 'tho', 'abd']])]
                if len(respiratory_effort_channels) > 0:
                    print(f'Success. Channel used: {respiratory_effort_channels}')
            if not sum(np.isin(respiratory_effort_channels, edf_channels)) > 0:
                print(f'Code cannot be performed: No effort belt channel found in the EDF file. \nThe file contains:{edf_channels}.')
                continue

            respiratory_effort_channels = [x for x in respiratory_effort_channels if x in edf_channels]

            signals = edf.get_data(picks=respiratory_effort_channels)  # signals.shape=(#channel, T)

            trace = np.mean(signals, axis=0) # Sum of ABD and CHEST effort belts.
            trace = trace[np.newaxis, :]

            # resample to 10Hz.
   
            ## filter and resample signal
            if notch_freq is not None and np.max(bandpass_freq)>=notch_freq:
                trace = notch_filter(trace, fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size+2padding)
            trace = filter_data(detrend(trace, axis=1), fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')  # take the value starting from *padding*, (#window, #ch, window_size+2padding)
            trace = resample(trace, up=new_fs, down=fs, axis=1) 
            fs = new_fs

            trace = trace.flatten()
            t = np.arange(len(trace))/fs # time array
            t = t.flatten()

            # This decommented section is only valid for the MGH .mat sleeplab files:
            # # path to respiration signal.
            # # data_path = 'trace.mat'
            # datadir = '';
            # data_path = os.path.join(datadir,'Signal_TwinData4_952.mat')
            # # sampling frequency
            # fs = 200
            # loads abdomen+chest respiration signal from an MGH matlab format file
            # print("loading data...")
            # t, trace = load_mgh_matlab_data(data_path, fs)

            # normalize respiration signal
            trace = clip_z_normalize(trace)

            # import pdb; pdb.set_trace()

            # remove outliers from signal based on signal envelope
            idx,trace = remove_outliers(t,trace,fs, min_height=0.01)
                
            # compute the upper and lower envelope of the respiration signal
            [up,lo]=create_env(trace, fs)
                
            # based on envelopes, central respiration signals can be detected
            apneas, hypopneas = detect_central_events(lo,up,fs)

            similarity_array = compute_similarity(up, lo, fs)

            apneas, hypopneas = post_processing_detections(apneas, hypopneas, similarity_array, fs)

            # saves report
            savepath_report = os.path.join(self_sim_results_dir, os.path.basename(edf_path).replace('.edf', '_report.csv'))
            report = generate_report(trace, apneas, hypopneas, fs)
            report.to_csv(savepath_report, index=False)

            # make the plots, saves as pdf and png:
            save_name = os.path.basename(edf_path).replace('.edf', '_figure')
            savepath_plot = os.path.join(self_sim_results_dir, save_name)
            # this curently creates a png and a .pdf
            plot_central_events(trace, apneas, hypopneas, savepath_plot)

            # save summary csv:
            try:
                path_hlg_summary = os.path.join(edf_folder, 'summary_HLG_percentage.csv')
                if os.path.exists(path_hlg_summary):
                    summary_hlg = pd.read_csv(path_hlg_summary)
                else:
                    summary_hlg = pd.DataFrame(columns = ['file', 'HLG_percentage (%)'])

                summary_hlg_tmp = pd.DataFrame(columns = ['file', 'HLG_percentage (%)'])
                summary_hlg_tmp.loc[0, 'file'] = os.path.basename(edf_path)

                hlg_perc = (sum(apneas) + sum(hypopneas)) / (len(trace) * 0.9) # small scaler because we consider HLG area also between two annotated apneas.

                summary_hlg_tmp.loc[0, 'HLG_percentage (%)'] = np.round(100*hlg_perc, 1)

                summary_hlg = pd.concat([summary_hlg, summary_hlg_tmp], axis=0, ignore_index=True)
                summary_hlg.drop_duplicates(inplace=True)

                summary_hlg.to_csv(path_hlg_summary, index=False)

            except Exception as e:
                print('Error in summary file computation/creation:')
                print(e)
                print('Continue with next file.')
                continue
            # print('runtime (sec):')
            # print(time.time() - starttime)

        except Exception as e:
            print(f'Error for {edf_path}:')
            print(e)
            print('Continue with next file.')

if __name__ == '__main__':
    main()


