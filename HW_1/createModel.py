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
EXP_NAME = DATA_PATH+"Yoel_2_raw.fif" ## file name to run the anaylsis on

features = ['app_entropy', 'decorr_time', 'higuchi_fd',
            'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility',
            'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
            'line_length', 'mean', 'ptp_amp', 'samp_entropy',
            'skewness', 'spect_edge_freq', 'spect_entropy', 'spect_slope',
            'std', 'svd_entropy', 'svd_fisher_info', 'teager_kaiser_energy',
            'variance', 'wavelet_coef_energy', 'zero_crossings', 'max_cross_corr',
            'nonlin_interdep', 'phase_lock_val', 'spect_corr', 'time_corr']

def preprocess():

    tmin, tmax = -1., 4.
    
    raw = mne.io.read_raw_fif(EXP_NAME, preload=True)
    
    raw.filter(7., 40., fir_design='firwin', skip_by_annotation='edge')
    
    events = mne.find_events(raw, 'STI')
    
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    
    
    event_id = {'Left': 1, 'right': 2,'none': 3}
    
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs.pick_types(eeg=True, exclude='bads')  # remove stim and EOG

    return epochs,raw

def trainCSP_LDA(epochs_data_train,labels,cv):
    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=14, reg=None, log=True, norm_trace=False)
    
    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    
    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

def train_mne_feature(data,labels,raw):
    pipe = Pipeline([('fe', FeatureExtractor(sfreq=raw.info['sfreq'],
                                         selected_funcs=["std","app_entropy",
                                                         "mean",
                                                         "kurtosis",
                                                         "max_cross_corr"])),
                 ('scaler', StandardScaler()),
                 ('clf', GradientBoostingClassifier())])
    skf = StratifiedKFold(n_splits=5, random_state=42)
    y = labels
    
    params_grid = {'fe__app_entropy__emb': np.arange(2, 5)}

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
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    
    # trainCSP_LDA(epochs_data_train,labels,cv)
    
    pipe = train_mne_feature(epochs_data_train,labels,raw)
    
    return pipe

if __name__ == '__main__':
    pipe = main()



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

