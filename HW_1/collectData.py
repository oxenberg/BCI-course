# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:24:18 2020

@author: oxenb
"""
import matplotlib.pyplot as plt
from mne.stats import bootstrap_confidence_interval
from mne.baseline import rescale

from pyOpenBCI import OpenBCICyton
import numpy as np
import mne
import random
import time

CH_AMOUNT = 16
TIME_BETWEEN_EVENTS = 3
SAMPLE_RATE = 125
TIME_BETWEEN_EVENTS_RATE = SAMPLE_RATE*TIME_BETWEEN_EVENTS

uVolts_per_count = (4500000)/24/(2**23-1) #uV/count


DATA_PATH = "data/"
EXP_NAME = DATA_PATH+"Or_3_raw.fif" #: give name to the expirement


EXPERIMENT_DURATION = 500
ITER = {"COUNT" : 0} #for cout the time 
ACTIONS = {1 : "LEFT",2 : "RIGHT",3 : "NONE"}

RUN_EXP = False #: to collect data change to true 

if RUN_EXP:
    board = OpenBCICyton(port='COM3', daisy=True)
start_time = time.time()
current_time = start_time
#########################

array_data = []
stim  = []


#: create the raw object from array

def create_raw_data(results,stim):
    ch_names = ['EEG ' + str(ID) for ID in range(CH_AMOUNT)]
    ch_type = 'eeg'
    info = mne.create_info(ch_names,SAMPLE_RATE,ch_type)
    rawData = mne.io.RawArray(results,info)
    
    #: add events data to raw
    stim_info = mne.create_info(['STI'], rawData.info['sfreq'], ['stim'])
    stim = np.expand_dims(stim, axis=0) 
    stim_raw = mne.io.RawArray(stim, stim_info)
    rawData.add_channels([stim_raw], force_update_info=True)
    
    
    
    #eventsData = mne.find_events(rawData, stim_channel='STI')
    return rawData


def run_expirement(sample):
    data = np.array(sample.channels_data)* uVolts_per_count
    
    all_time = time.time() - start_time
    ITER["COUNT"] +=1
    if ITER["COUNT"]% TIME_BETWEEN_EVENTS_RATE == 0 :
         int_action = random.randint(1, 3)
         print(ACTIONS[int_action])
         stim.append(int_action)
    else:
        stim.append(0)
    array_data.append(data)
    
    # print((all_time,event_time) )
    
    if int(all_time) >= EXPERIMENT_DURATION:
        board.stop_stream()
    
def start_expirement():
    
    board.start_stream(run_expirement)
    board.disconnect()


    
if RUN_EXP:
    start_expirement()
    
    
    ##data exploration
    #: transform to array format
    array_data = np.array(array_data)
    array_data = array_data.transpose()
    array_data_v = array_data* 10**(-6) #: to volt
    
    #: transform to raw (mne) format
    rawData = create_raw_data(array_data_v,stim)
    
    #: filter electrecy 50 hz freq
    rawData = rawData.filter(None, 45., fir_design='firwin')
    
    #: mainualy filtering
    events = mne.find_events(rawData, stim_channel='STI')
    
    
    annot_from_events = mne.annotations_from_events(
        events=events, event_desc=ACTIONS, sfreq=rawData.info['sfreq'])
    rawData.set_annotations(annot_from_events)
    
    rawData.plot()
    
    #: save data after cleaning
    rawData.save(EXP_NAME,overwrite=True)

else:
    rawData = mne.io.read_raw_fif(EXP_NAME, preload=True)
    
    events = mne.find_events(rawData, stim_channel='STI')

    event_dict = {'LEFT': 1, 'RIGHT': 2, 'NONE': 3}
    
    
    rawData.plot_psd(fmax=50,spatial_colors = True)
    
    
    fig = mne.viz.plot_events(events,event_id  = event_dict, sfreq=rawData.info['sfreq'],
                              first_samp=rawData.first_samp)
    
    reject_criteria = dict(eeg=150e-6)       # 250 µV
    
    
    epochs = mne.Epochs(rawData, events, event_id=event_dict, tmin=-0.1, tmax=0.2,preload=True)
    
    left_epochs = epochs['LEFT']
    right_epochs = epochs['RIGHT']
    none_epochs = epochs['NONE']
    
    
    left_epochs = left_epochs.average()
    right_epochs = right_epochs.average()
    none_epochs = none_epochs.average()
    
    mne.viz.plot_compare_evokeds(dict(left=left_epochs, right=right_epochs,nothing = none_epochs),
                                  legend='upper left', show_sensors='upper right')
    
    

    epochs.plot_image(combine='mean')
    event_id, tmin, tmax = 1, -0.5, 3.
    baseline = None
    
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)
    ]
    
    frequency_map = list()
    
    for band, fmin, fmax in iter_freqs:
        # bandpass filter
        raw = rawData.copy()
        
        raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.
    
        # epoch
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            preload=True)
        # remove evoked response
        epochs.subtract_evoked()
    
        # get analytic signal (envelope)
        epochs.apply_hilbert(envelope=True)
        frequency_map.append(((band, fmin, fmax), epochs.average()))
    
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
    for ((freq_name, fmin, fmax), average), color, ax in zip(
            frequency_map, colors, axes.ravel()[::-1]):
        times = average.times * 1e3
        gfp = np.sum(average.data ** 2, axis=0)
        gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
        ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
        ax.axhline(0, linestyle='--', color='grey', linewidth=2)
        ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                      stat_fun=stat_fun)
        ci_low = rescale(ci_low, average.times, baseline=(None, 0))
        ci_up = rescale(ci_up, average.times, baseline=(None, 0))
        ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
        ax.grid(True)
        ax.set_ylabel('GFP')
        ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                    xy=(0.95, 0.8),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.set_xlim(-500, 3000)
    
    axes.ravel()[-1].set_xlabel('Time [ms]')
#########################







    
    
    
    

