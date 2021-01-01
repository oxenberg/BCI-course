# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:38:29 2020

@author: oxenb
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mne_features.feature_extraction import FeatureExtractor

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     StratifiedKFold)

from mne import Epochs, pick_types, events_from_annotations
import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

DATA_PATH = "data/"
EXP_NAME = DATA_PATH+"Or_3_raw.fif" ## file name to run the anaylsis on

features = ['app_entropy', 'decorr_time', 'higuchi_fd',
            'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility',
            'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
            'line_length', 'mean', 'ptp_amp', 'samp_entropy',
            'skewness', 'spect_edge_freq', 'spect_entropy', 'spect_slope',
            'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy',
            'variance', 'wavelet_coef_energy', 'zero_crossings', 'max_cross_corr',
            'nonlin_interdep', 'phase_lock_val', 'spect_corr', 'time_corr']

selected_features = ["std","mean","kurtosis","skewness"] # can be cgahnged to any feature


def preprocess():

    tmin, tmax = -1., 0.8 #: need to check the best
    
    raw = mne.io.read_raw_fif(EXP_NAME, preload=True)
    
    raw.filter(5., 40., fir_design='firwin', skip_by_annotation='edge')
    
    events = mne.find_events(raw, 'STI')
    
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    
    event_id = {'Left': 1, 'right': 2,'none': 3}
    
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs.pick_types(eeg=True, exclude='bads')  # remove stim and EOG

    return epochs,raw


def train_mne_feature(data,labels,raw):
    pipe = Pipeline([('fe', FeatureExtractor(sfreq = raw.info['sfreq'],
                                         selected_funcs = selected_features)),
                 ('scaler', StandardScaler()),
                 ('clf', GradientBoostingClassifier())])
    y = labels
    
    # params_grid = {'fe__app_entropy__emb': np.arange(2, 5)} #: can addd gradinet boost hyperparametrs
    params_grid = {} #: can addd gradinet boost hyperparametrs

    gs = GridSearchCV(estimator=pipe, param_grid=params_grid,
                      cv=StratifiedKFold(n_splits=5, random_state=42), n_jobs=1,
                      return_train_score=True)
    gs.fit(data, y)
    
    
    scores = pd.DataFrame(gs.cv_results_)
    print(scores[['params', 'mean_test_score', 'mean_train_score']])
    # Best parameters obtained with GridSearchCV:
    print(gs.best_params_)
    
    
    #: run the best model maybe need to create test seprate dataset
    # gs_best = gs.best_estimator_
    # new_scores = cross_val_score(gs_best, data, y, cv=skf)

    # print('Cross-validation accuracy score (with optimized parameters) = %1.3f '
    #       '(+/- %1.5f)' % (np.mean(new_scores), np.std(new_scores)))
    
    return pipe

    
    

def main():
    epochs,raw =  preprocess()
    
    labels = epochs.events[:, -1]

    # get MEG and EEG data
    epochs_data_train = epochs.get_data()
            
    pipe = train_mne_feature(epochs_data_train,labels,raw)
    
    transformed_data = pipe["fe"].fit_transform(epochs_data_train) #: transformed_data is matrix dim by the featuhers X events
    
    
    return pipe,epochs_data_train

if __name__ == '__main__':
    pipe,epochs_data_train = main()
    



'''
['app_entropy', 'decorr_time', 'energy_freq_bands', 'higuchi_fd',
 'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility'
 'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis', 'line_length',
 'mean', 'pow_freq_bands', 'ptp_amp', 'samp_entropy', 'skewness', 
 'spect_edge_freq', 'spect_entropy', 'spect_slope', 'std', 'svd_entropy',
 'svd_fisher_info', 'teager_kaiser_energy', 'variance', 'wavelet_coef_energy',
 'zero_crossings', 'max_cross_corr', 'nonlin_interdep', 'phase_lock_val',
 'spect_corr', 'time_corr']
'''

