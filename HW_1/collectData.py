# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 20:24:18 2020

@author: oxenb
"""

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
EXP_NAME = DATA_PATH+"Or_2_raw.fif" #: give name to the expirement


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
    
    
    rawData.plot_psd(fmax=50)
    
    
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


#########################







    
    
    
    

