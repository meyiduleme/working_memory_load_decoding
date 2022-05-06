#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020 03 20

@author: Gérard
"""

# =============================================================================
# EEGLAB file to CSV file
# =============================================================================

from mne.io import read_epochs_eeglab
import numpy as np
from functions import *
from os import makedirs
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_data", type=str, default="./data/")
parser.add_argument("-o", "--output_folder", type=str, default="./results/")
parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["01", "02", "03", "04", "05", "06", "07", "09", "11"])
parser.add_argument("-l", "--lobes", nargs="+", type=str, default=["components_trials"])
parser.add_argument("-a", "--averaging_values",  nargs="+", type=int, default=[4])
parser.add_argument("-m", "--morlet_freqs", nargs="+", type=int, default=[6, 10, 25, 50])

args = parser.parse_args()

nb_trial = args.averaging_values[0]    

lobe = args.lobes[0]
results_folder = args.output_folder+lobe+"/"

for subject in args.list_subject:
    
    #_________________________data import______________________________#

    file = read_epochs_eeglab(input_fname = args.input_data + "Subject_" + subject + "_Day_02_EEG_RAW_RS_BP_EP_BL_ICA_RJ_"+lobe+".set")

    dim1 = file.get_data().shape[0]
    dim2 = file.get_data().shape[1]
    dim3 = file.get_data().shape[2]

    x = file.get_data()
    y = file.events[:,2]

    x, time = cut_time(x, file)    
    print("data_shape: ", x.shape)

    #_________________________RAW DATA______________________________#
    x_raw = np.reshape( x, (x.shape[0], x.shape[1]*x.shape[2]) )
    print("x_raw_data_shape: ", x_raw.shape)
    y_raw = np.copy(y)
    for i in range(len(y)):
        if y[i] == 1 or y[i] == 2:
            y_raw[i] = 2
        elif y[i] == 3 or y[i] == 4:
            y_raw[i] = 3
        elif y[i] == 5 or y[i] == 6:
            y_raw[i] = 1
        else:
            print("Class error at index : ", i)            
    print("y_raw_data_shape: ", y_raw.shape)
    xy_raw = np.hstack((x_raw, y_raw.reshape(len(y_raw),1)))
    print("xy_raw_data_shape: ", xy_raw.shape)

    #séparation par classes
    x1_raw, x2_raw, x3_raw = [], [], []
    y1_raw, y2_raw, y3_raw = [], [], []    
    for i in range(len(y)):
        if y[i] == 1 or y[i] == 2:
            x2_raw.append(x[i])
            y2_raw.append(2)
        elif y[i] == 3 or y[i] == 4:
            x3_raw.append(x[i])
            y3_raw.append(3)
        elif y[i] == 5 or y[i] == 6:
            x1_raw.append(x[i])
            y1_raw.append(1)
        else:
            print("Class error at index : ", i)    
    
    # Moyennage
    x1_raw_mean, x2_raw_mean, x3_raw_mean = [], [], []
    y1_raw_mean, y2_raw_mean, y3_raw_mean = [], [], []
    for i in range(len(x1_raw) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x1_raw[nb_trial * i + j])
        x1_raw_mean.append(np.mean(list_av, axis = 0))
        y1_raw_mean.append(1)
    for i in range(len(x2_raw) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x2_raw[nb_trial * i + j])
        x2_raw_mean.append(np.mean(list_av, axis = 0))
        y2_raw_mean.append(2)
    for i in range(len(x3_raw) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x3_raw[nb_trial * i + j])
        x3_raw_mean.append(np.mean(list_av, axis = 0))
        y3_raw_mean.append(3)    

    #__________________Creation ndarray 12________________#
    x12_raw_3d = np.array(x1_raw + x2_raw)
    x12_raw_3d = np.ndarray(shape=(len(x12_raw_3d), len(x12_raw_3d[0]), len(x12_raw_3d[0][0])), buffer = x12_raw_3d)    
    x12_raw_2d = np.reshape( x12_raw_3d, (x12_raw_3d.shape[0], x12_raw_3d.shape[1]*x12_raw_3d.shape[2]) )
    y12_raw = np.ndarray(shape=(len(y1_raw+y2_raw)), buffer=np.array(y1_raw+y2_raw), dtype=int)
    xy12_raw = np.hstack((x12_raw_2d, y12_raw.reshape(len(y12_raw),1)))

    x12_raw_mean_3d = np.array(x1_raw_mean + x2_raw_mean)
    x12_raw_mean_3d = np.ndarray(shape=(len(x12_raw_mean_3d), len(x12_raw_mean_3d[0]), len(x12_raw_mean_3d[0][0])), buffer = x12_raw_mean_3d)    
    x12_raw_mean_2d = np.reshape( x12_raw_mean_3d, (x12_raw_mean_3d.shape[0], x12_raw_mean_3d.shape[1]*x12_raw_mean_3d.shape[2]) )
    y12_raw_mean = np.ndarray(shape=(len(y1_raw_mean+y2_raw_mean)), buffer=np.array(y1_raw_mean+y2_raw_mean), dtype=int)
    xy12_raw_mean = np.hstack((x12_raw_mean_2d, y12_raw_mean.reshape(len(y12_raw_mean),1)))

    x12_raw_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x12_raw_mean_3d, y12_raw_mean)     
    x12_raw_mean_CSP_2d = np.reshape( x12_raw_mean_CSP_3d, (x12_raw_mean_CSP_3d.shape[0], x12_raw_mean_CSP_3d.shape[1]*x12_raw_mean_CSP_3d.shape[2]) )
    y12_raw_mean_CSP = np.copy(y12_raw_mean)
    xy12_raw_mean_CSP = np.hstack((x12_raw_mean_CSP_2d, y12_raw_mean_CSP.reshape(len(y12_raw_mean_CSP),1)))

    x12_raw_mean_CSP_LogVar = np.zeros((x12_raw_mean_CSP_3d.shape[0] , x12_raw_mean_CSP_3d.shape[1]))
    for iEpoch in range(x12_raw_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x12_raw_mean_CSP_3d.shape[1])
        for iCSP in range(x12_raw_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x12_raw_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x12_raw_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y12_raw_mean_CSP_LogVar = np.copy(y12_raw_mean)
    xy12_raw_mean_CSP_LogVar = np.hstack((x12_raw_mean_CSP_LogVar, y12_raw_mean_CSP.reshape(len(y12_raw_mean_CSP),1)))

    #__________________Creation ndarray 13________________#
    x13_raw_3d = np.array(x1_raw + x3_raw)
    x13_raw_3d = np.ndarray(shape=(len(x13_raw_3d), len(x13_raw_3d[0]), len(x13_raw_3d[0][0])), buffer = x13_raw_3d)    
    x13_raw_2d = np.reshape( x13_raw_3d, (x13_raw_3d.shape[0], x13_raw_3d.shape[1]*x13_raw_3d.shape[2]) )
    y13_raw = np.ndarray(shape=(len(y1_raw+y3_raw)), buffer=np.array(y1_raw+y3_raw), dtype=int)
    xy13_raw = np.hstack((x13_raw_2d, y13_raw.reshape(len(y13_raw),1)))

    x13_raw_mean_3d = np.array(x1_raw_mean + x3_raw_mean)
    x13_raw_mean_3d = np.ndarray(shape=(len(x13_raw_mean_3d), len(x13_raw_mean_3d[0]), len(x13_raw_mean_3d[0][0])), buffer = x13_raw_mean_3d)    
    x13_raw_mean_2d = np.reshape( x13_raw_mean_3d, (x13_raw_mean_3d.shape[0], x13_raw_mean_3d.shape[1]*x13_raw_mean_3d.shape[2]) )
    y13_raw_mean = np.ndarray(shape=(len(y1_raw_mean+y3_raw_mean)), buffer=np.array(y1_raw_mean+y3_raw_mean), dtype=int)
    xy13_raw_mean = np.hstack((x13_raw_mean_2d, y13_raw_mean.reshape(len(y13_raw_mean),1)))

    x13_raw_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x13_raw_mean_3d, y13_raw_mean)     
    x13_raw_mean_CSP_2d = np.reshape( x13_raw_mean_CSP_3d, (x13_raw_mean_CSP_3d.shape[0], x13_raw_mean_CSP_3d.shape[1]*x13_raw_mean_CSP_3d.shape[2]) )
    y13_raw_mean_CSP = np.copy(y13_raw_mean)
    xy13_raw_mean_CSP = np.hstack((x13_raw_mean_CSP_2d, y13_raw_mean_CSP.reshape(len(y13_raw_mean_CSP),1)))

    x13_raw_mean_CSP_LogVar = np.zeros((x13_raw_mean_CSP_3d.shape[0] , x13_raw_mean_CSP_3d.shape[1]))
    for iEpoch in range(x13_raw_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x13_raw_mean_CSP_3d.shape[1])
        for iCSP in range(x13_raw_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x13_raw_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x13_raw_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y13_raw_mean_CSP_LogVar = np.copy(y13_raw_mean)
    xy13_raw_mean_CSP_LogVar = np.hstack((x13_raw_mean_CSP_LogVar, y13_raw_mean_CSP.reshape(len(y13_raw_mean_CSP),1)))

    #__________________Creation ndarray 23________________#
    x23_raw_3d = np.array(x2_raw + x3_raw)
    x23_raw_3d = np.ndarray(shape=(len(x23_raw_3d), len(x23_raw_3d[0]), len(x23_raw_3d[0][0])), buffer = x23_raw_3d)    
    x23_raw_2d = np.reshape( x23_raw_3d, (x23_raw_3d.shape[0], x23_raw_3d.shape[1]*x23_raw_3d.shape[2]) )
    y23_raw = np.ndarray(shape=(len(y2_raw+y3_raw)), buffer=np.array(y2_raw+y3_raw), dtype=int)
    xy23_raw = np.hstack((x23_raw_2d, y23_raw.reshape(len(y23_raw),1)))

    x23_raw_mean_3d = np.array(x2_raw_mean + x3_raw_mean)
    x23_raw_mean_3d = np.ndarray(shape=(len(x23_raw_mean_3d), len(x23_raw_mean_3d[0]), len(x23_raw_mean_3d[0][0])), buffer = x23_raw_mean_3d)    
    x23_raw_mean_2d = np.reshape( x23_raw_mean_3d, (x23_raw_mean_3d.shape[0], x23_raw_mean_3d.shape[1]*x23_raw_mean_3d.shape[2]) )
    y23_raw_mean = np.ndarray(shape=(len(y2_raw_mean+y3_raw_mean)), buffer=np.array(y2_raw_mean+y3_raw_mean), dtype=int)
    xy23_raw_mean = np.hstack((x23_raw_mean_2d, y23_raw_mean.reshape(len(y23_raw_mean),1)))

    x23_raw_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x23_raw_mean_3d, y23_raw_mean)     
    x23_raw_mean_CSP_2d = np.reshape( x23_raw_mean_CSP_3d, (x23_raw_mean_CSP_3d.shape[0], x23_raw_mean_CSP_3d.shape[1]*x23_raw_mean_CSP_3d.shape[2]) )
    y23_raw_mean_CSP = np.copy(y23_raw_mean)
    xy23_raw_mean_CSP = np.hstack((x23_raw_mean_CSP_2d, y23_raw_mean_CSP.reshape(len(y23_raw_mean_CSP),1)))

    x23_raw_mean_CSP_LogVar = np.zeros((x23_raw_mean_CSP_3d.shape[0] , x23_raw_mean_CSP_3d.shape[1]))
    for iEpoch in range(x23_raw_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x23_raw_mean_CSP_3d.shape[1])
        for iCSP in range(x23_raw_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x23_raw_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x23_raw_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y23_raw_mean_CSP_LogVar = np.copy(y23_raw_mean)
    xy23_raw_mean_CSP_LogVar = np.hstack((x23_raw_mean_CSP_LogVar, y23_raw_mean_CSP.reshape(len(y23_raw_mean_CSP),1)))
 
    
    #__________________Creation ndarray 123________________#    
    x123_raw_3d = np.array(x1_raw + x2_raw + x3_raw)
    x123_raw_3d = np.ndarray(shape=(len(x123_raw_3d), len(x123_raw_3d[0]), len(x123_raw_3d[0][0])), buffer = x123_raw_3d)    
    x123_raw_2d = np.reshape( x123_raw_3d, (x123_raw_3d.shape[0], x123_raw_3d.shape[1]*x123_raw_3d.shape[2]) )
    y123_raw = np.ndarray(shape=(len(y1_raw+y2_raw+y3_raw)), buffer=np.array(y1_raw+y2_raw+y3_raw), dtype=int)
    xy123_raw = np.hstack((x123_raw_2d, y123_raw.reshape(len(y123_raw),1)))

    x123_raw_mean_3d = np.array(x1_raw_mean + x2_raw_mean + x3_raw_mean)
    x123_raw_mean_3d = np.ndarray(shape=(len(x123_raw_mean_3d), len(x123_raw_mean_3d[0]), len(x123_raw_mean_3d[0][0])), buffer = x123_raw_mean_3d)    
    x123_raw_mean_2d = np.reshape( x123_raw_mean_3d, (x123_raw_mean_3d.shape[0], x123_raw_mean_3d.shape[1]*x123_raw_mean_3d.shape[2]) )
    y123_raw_mean = np.ndarray(shape=(len(y1_raw_mean+y2_raw_mean+y3_raw_mean)), buffer=np.array(y1_raw_mean+y2_raw_mean+y3_raw_mean), dtype=int)
    xy123_raw_mean = np.hstack((x123_raw_mean_2d, y123_raw_mean.reshape(len(y123_raw_mean),1)))

    x123_raw_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x123_raw_mean_3d, y123_raw_mean)     
    x123_raw_mean_CSP_2d = np.reshape( x123_raw_mean_CSP_3d, (x123_raw_mean_CSP_3d.shape[0], x123_raw_mean_CSP_3d.shape[1]*x123_raw_mean_CSP_3d.shape[2]) )
    y123_raw_mean_CSP = np.copy(y123_raw_mean)
    xy123_raw_mean_CSP = np.hstack((x123_raw_mean_CSP_2d, y123_raw_mean_CSP.reshape(len(y123_raw_mean_CSP),1)))

    x123_raw_mean_CSP_LogVar = np.zeros((x123_raw_mean_CSP_3d.shape[0] , x123_raw_mean_CSP_3d.shape[1]))
    for iEpoch in range(x123_raw_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x123_raw_mean_CSP_3d.shape[1])
        for iCSP in range(x123_raw_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x123_raw_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x123_raw_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y123_raw_mean_CSP_LogVar = np.copy(y123_raw_mean)
    xy123_raw_mean_CSP_LogVar = np.hstack((x123_raw_mean_CSP_LogVar, y123_raw_mean_CSP.reshape(len(y123_raw_mean_CSP),1)))



    #_________________________MORLET DATA______________________________# 
    xx = morlet_tfr( x, file.info['sfreq'], args.morlet_freqs )
    xx = np.reshape( xx, (xx.shape[0] , xx.shape[1] * xx.shape[2] ,  xx.shape[3]) )
    print("xx_data_shape: ", xx.shape, "\n")
    yy = np.copy(y)
    print("yy_data_shape: ", yy.shape, "\n")
    # xxyy = np.hstack((xx, yy.reshape(len(yy),1)))
    # print("xxyy_morlet_data_shape: ", xxyy.shape, "\n")
        
    x_morlet = np.reshape( xx, (xx.shape[0], xx.shape[1]*xx.shape[2]) )
    print("x_morlet_data_shape: ", x_morlet.shape)
    y_morlet = np.copy(yy)
    for i in range(len(yy)):
        if yy[i] == 1 or yy[i] == 2:
            y_morlet[i] = 2
        elif yy[i] == 3 or yy[i] == 4:
            y_morlet[i] = 3
        elif yy[i] == 5 or yy[i] == 6:
            y_morlet[i] = 1
        else:
            print("Class error at index : ", i)            
    print("y_morlet_data_shape: ", y_morlet.shape)
    xy_morlet = np.hstack((x_morlet, y_morlet.reshape(len(y_morlet),1)))
    print("xy_morlet_data_shape: ", xy_morlet.shape)

    #séparation par classes
    x1_morlet, x2_morlet, x3_morlet = [], [], []
    y1_morlet, y2_morlet, y3_morlet = [], [], []    
    for i in range(len(y)):
        if yy[i] == 1 or yy[i] == 2:
            x2_morlet.append(xx[i])
            y2_morlet.append(2)
        elif yy[i] == 3 or yy[i] == 4:
            x3_morlet.append(xx[i])
            y3_morlet.append(3)
        elif yy[i] == 5 or yy[i] == 6:
            x1_morlet.append(xx[i])
            y1_morlet.append(1)
        else:
            print("Class error at index : ", i)    
    
    # Moyennage
    x1_morlet_mean, x2_morlet_mean, x3_morlet_mean = [], [], []
    y1_morlet_mean, y2_morlet_mean, y3_morlet_mean = [], [], []
    for i in range(len(x1_morlet) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x1_morlet[nb_trial * i + j])
        x1_morlet_mean.append(np.mean(list_av, axis = 0))
        y1_morlet_mean.append(1)
    for i in range(len(x2_morlet) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x2_morlet[nb_trial * i + j])
        x2_morlet_mean.append(np.mean(list_av, axis = 0))
        y2_morlet_mean.append(2)
    for i in range(len(x3_morlet) // nb_trial):
        list_av = []
        for j in range(nb_trial):
            list_av.append(x3_morlet[nb_trial * i + j])
        x3_morlet_mean.append(np.mean(list_av, axis = 0))
        y3_morlet_mean.append(3)    

    #__________________Creation ndarray 12________________#
    x12_morlet_3d = np.array(x1_morlet + x2_morlet)
    x12_morlet_3d = np.ndarray(shape=(len(x12_morlet_3d), len(x12_morlet_3d[0]), len(x12_morlet_3d[0][0])), buffer = x12_morlet_3d)    
    x12_morlet_2d = np.reshape( x12_morlet_3d, (x12_morlet_3d.shape[0], x12_morlet_3d.shape[1]*x12_morlet_3d.shape[2]) )
    y12_morlet = np.ndarray(shape=(len(y1_morlet+y2_morlet)), buffer=np.array(y1_morlet+y2_morlet), dtype=int)
    xy12_morlet = np.hstack((x12_morlet_2d, y12_morlet.reshape(len(y12_morlet),1)))

    x12_morlet_mean_3d = np.array(x1_morlet_mean + x2_morlet_mean)
    x12_morlet_mean_3d = np.ndarray(shape=(len(x12_morlet_mean_3d), len(x12_morlet_mean_3d[0]), len(x12_morlet_mean_3d[0][0])), buffer = x12_morlet_mean_3d)    
    x12_morlet_mean_2d = np.reshape( x12_morlet_mean_3d, (x12_morlet_mean_3d.shape[0], x12_morlet_mean_3d.shape[1]*x12_morlet_mean_3d.shape[2]) )
    y12_morlet_mean = np.ndarray(shape=(len(y1_morlet_mean+y2_morlet_mean)), buffer=np.array(y1_morlet_mean+y2_morlet_mean), dtype=int)
    xy12_morlet_mean = np.hstack((x12_morlet_mean_2d, y12_morlet_mean.reshape(len(y12_morlet_mean),1)))

    x12_morlet_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x12_morlet_mean_3d, y12_morlet_mean)     
    x12_morlet_mean_CSP_2d = np.reshape( x12_morlet_mean_CSP_3d, (x12_morlet_mean_CSP_3d.shape[0], x12_morlet_mean_CSP_3d.shape[1]*x12_morlet_mean_CSP_3d.shape[2]) )
    y12_morlet_mean_CSP = np.copy(y12_morlet_mean)
    xy12_morlet_mean_CSP = np.hstack((x12_morlet_mean_CSP_2d, y12_morlet_mean_CSP.reshape(len(y12_morlet_mean_CSP),1)))

    x12_morlet_mean_CSP_LogVar = np.zeros((x12_morlet_mean_CSP_3d.shape[0] , x12_morlet_mean_CSP_3d.shape[1]))
    for iEpoch in range(x12_morlet_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x12_morlet_mean_CSP_3d.shape[1])
        for iCSP in range(x12_morlet_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x12_morlet_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x12_morlet_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y12_morlet_mean_CSP_LogVar = np.copy(y12_morlet_mean)
    xy12_morlet_mean_CSP_LogVar = np.hstack((x12_morlet_mean_CSP_LogVar, y12_morlet_mean_CSP.reshape(len(y12_morlet_mean_CSP),1)))

    #__________________Creation ndarray 13________________#
    x13_morlet_3d = np.array(x1_morlet + x3_morlet)
    x13_morlet_3d = np.ndarray(shape=(len(x13_morlet_3d), len(x13_morlet_3d[0]), len(x13_morlet_3d[0][0])), buffer = x13_morlet_3d)    
    x13_morlet_2d = np.reshape( x13_morlet_3d, (x13_morlet_3d.shape[0], x13_morlet_3d.shape[1]*x13_morlet_3d.shape[2]) )
    y13_morlet = np.ndarray(shape=(len(y1_morlet+y3_morlet)), buffer=np.array(y1_morlet+y3_morlet), dtype=int)
    xy13_morlet = np.hstack((x13_morlet_2d, y13_morlet.reshape(len(y13_morlet),1)))

    x13_morlet_mean_3d = np.array(x1_morlet_mean + x3_morlet_mean)
    x13_morlet_mean_3d = np.ndarray(shape=(len(x13_morlet_mean_3d), len(x13_morlet_mean_3d[0]), len(x13_morlet_mean_3d[0][0])), buffer = x13_morlet_mean_3d)    
    x13_morlet_mean_2d = np.reshape( x13_morlet_mean_3d, (x13_morlet_mean_3d.shape[0], x13_morlet_mean_3d.shape[1]*x13_morlet_mean_3d.shape[2]) )
    y13_morlet_mean = np.ndarray(shape=(len(y1_morlet_mean+y3_morlet_mean)), buffer=np.array(y1_morlet_mean+y3_morlet_mean), dtype=int)
    xy13_morlet_mean = np.hstack((x13_morlet_mean_2d, y13_morlet_mean.reshape(len(y13_morlet_mean),1)))

    x13_morlet_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x13_morlet_mean_3d, y13_morlet_mean)     
    x13_morlet_mean_CSP_2d = np.reshape( x13_morlet_mean_CSP_3d, (x13_morlet_mean_CSP_3d.shape[0], x13_morlet_mean_CSP_3d.shape[1]*x13_morlet_mean_CSP_3d.shape[2]) )
    y13_morlet_mean_CSP = np.copy(y13_morlet_mean)
    xy13_morlet_mean_CSP = np.hstack((x13_morlet_mean_CSP_2d, y13_morlet_mean_CSP.reshape(len(y13_morlet_mean_CSP),1)))

    x13_morlet_mean_CSP_LogVar = np.zeros((x13_morlet_mean_CSP_3d.shape[0] , x13_morlet_mean_CSP_3d.shape[1]))
    for iEpoch in range(x13_morlet_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x13_morlet_mean_CSP_3d.shape[1])
        for iCSP in range(x13_morlet_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x13_morlet_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x13_morlet_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y13_morlet_mean_CSP_LogVar = np.copy(y13_morlet_mean)
    xy13_morlet_mean_CSP_LogVar = np.hstack((x13_morlet_mean_CSP_LogVar, y13_morlet_mean_CSP.reshape(len(y13_morlet_mean_CSP),1)))

    #__________________Creation ndarray 23________________#
    x23_morlet_3d = np.array(x2_morlet + x3_morlet)
    x23_morlet_3d = np.ndarray(shape=(len(x23_morlet_3d), len(x23_morlet_3d[0]), len(x23_morlet_3d[0][0])), buffer = x23_morlet_3d)    
    x23_morlet_2d = np.reshape( x23_morlet_3d, (x23_morlet_3d.shape[0], x23_morlet_3d.shape[1]*x23_morlet_3d.shape[2]) )
    y23_morlet = np.ndarray(shape=(len(y2_morlet+y3_morlet)), buffer=np.array(y2_morlet+y3_morlet), dtype=int)
    xy23_morlet = np.hstack((x23_morlet_2d, y23_morlet.reshape(len(y23_morlet),1)))

    x23_morlet_mean_3d = np.array(x2_morlet_mean + x3_morlet_mean)
    x23_morlet_mean_3d = np.ndarray(shape=(len(x23_morlet_mean_3d), len(x23_morlet_mean_3d[0]), len(x23_morlet_mean_3d[0][0])), buffer = x23_morlet_mean_3d)    
    x23_morlet_mean_2d = np.reshape( x23_morlet_mean_3d, (x23_morlet_mean_3d.shape[0], x23_morlet_mean_3d.shape[1]*x23_morlet_mean_3d.shape[2]) )
    y23_morlet_mean = np.ndarray(shape=(len(y2_morlet_mean+y3_morlet_mean)), buffer=np.array(y2_morlet_mean+y3_morlet_mean), dtype=int)
    xy23_morlet_mean = np.hstack((x23_morlet_mean_2d, y23_morlet_mean.reshape(len(y23_morlet_mean),1)))

    x23_morlet_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x23_morlet_mean_3d, y23_morlet_mean)     
    x23_morlet_mean_CSP_2d = np.reshape( x23_morlet_mean_CSP_3d, (x23_morlet_mean_CSP_3d.shape[0], x23_morlet_mean_CSP_3d.shape[1]*x23_morlet_mean_CSP_3d.shape[2]) )
    y23_morlet_mean_CSP = np.copy(y23_morlet_mean)
    xy23_morlet_mean_CSP = np.hstack((x23_morlet_mean_CSP_2d, y23_morlet_mean_CSP.reshape(len(y23_morlet_mean_CSP),1)))

    x23_morlet_mean_CSP_LogVar = np.zeros((x23_morlet_mean_CSP_3d.shape[0] , x23_morlet_mean_CSP_3d.shape[1]))
    for iEpoch in range(x23_morlet_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x23_morlet_mean_CSP_3d.shape[1])
        for iCSP in range(x23_morlet_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x23_morlet_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x23_morlet_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y23_morlet_mean_CSP_LogVar = np.copy(y23_morlet_mean)
    xy23_morlet_mean_CSP_LogVar = np.hstack((x23_morlet_mean_CSP_LogVar, y23_morlet_mean_CSP.reshape(len(y23_morlet_mean_CSP),1)))

    #__________________Creation ndarray 123________________#    
    x123_morlet_3d = np.array(x1_morlet + x2_morlet + x3_morlet)
    x123_morlet_3d = np.ndarray(shape=(len(x123_morlet_3d), len(x123_morlet_3d[0]), len(x123_morlet_3d[0][0])), buffer = x123_morlet_3d)    
    x123_morlet_2d = np.reshape( x123_morlet_3d, (x123_morlet_3d.shape[0], x123_morlet_3d.shape[1]*x123_morlet_3d.shape[2]) )
    y123_morlet = np.ndarray(shape=(len(y1_morlet+y2_morlet+y3_morlet)), buffer=np.array(y1_morlet+y2_morlet+y3_morlet), dtype=int)
    xy123_morlet = np.hstack((x123_morlet_2d, y123_morlet.reshape(len(y123_morlet),1)))

    x123_morlet_mean_3d = np.array(x1_morlet_mean + x2_morlet_mean + x3_morlet_mean)
    x123_morlet_mean_3d = np.ndarray(shape=(len(x123_morlet_mean_3d), len(x123_morlet_mean_3d[0]), len(x123_morlet_mean_3d[0][0])), buffer = x123_morlet_mean_3d)    
    x123_morlet_mean_2d = np.reshape( x123_morlet_mean_3d, (x123_morlet_mean_3d.shape[0], x123_morlet_mean_3d.shape[1]*x123_morlet_mean_3d.shape[2]) )
    y123_morlet_mean = np.ndarray(shape=(len(y1_morlet_mean+y2_morlet_mean+y3_morlet_mean)), buffer=np.array(y1_morlet_mean+y2_morlet_mean+y3_morlet_mean), dtype=int)
    xy123_morlet_mean = np.hstack((x123_morlet_mean_2d, y123_morlet_mean.reshape(len(y123_morlet_mean),1)))

    x123_morlet_mean_CSP_3d = CSP(n_components=6, transform_into='csp_space').fit_transform(x123_morlet_mean_3d, y123_morlet_mean)     
    x123_morlet_mean_CSP_2d = np.reshape( x123_morlet_mean_CSP_3d, (x123_morlet_mean_CSP_3d.shape[0], x123_morlet_mean_CSP_3d.shape[1]*x123_morlet_mean_CSP_3d.shape[2]) )
    y123_morlet_mean_CSP = np.copy(y123_morlet_mean)
    xy123_morlet_mean_CSP = np.hstack((x123_morlet_mean_CSP_2d, y123_morlet_mean_CSP.reshape(len(y123_morlet_mean_CSP),1)))

    x123_morlet_mean_CSP_LogVar = np.zeros((x123_morlet_mean_CSP_3d.shape[0] , x123_morlet_mean_CSP_3d.shape[1]))
    for iEpoch in range(x123_morlet_mean_CSP_3d.shape[0]):
        logVarVector = np.zeros(x123_morlet_mean_CSP_3d.shape[1])
        for iCSP in range(x123_morlet_mean_CSP_3d.shape[1]):
            logVarVector[iCSP] = np.var(x123_morlet_mean_CSP_3d[iEpoch,iCSP,:])
        logVarVector = np.log(logVarVector / np.sum(logVarVector))
        x123_morlet_mean_CSP_LogVar[iEpoch,:] = np.transpose(logVarVector)     
    y123_morlet_mean_CSP_LogVar = np.copy(y123_morlet_mean)
    xy123_morlet_mean_CSP_LogVar = np.hstack((x123_morlet_mean_CSP_LogVar, y123_morlet_mean_CSP.reshape(len(y123_morlet_mean_CSP),1)))

    #_________________________SAVE FILES______________________________# 
    makedirs(results_folder+"datacsv", exist_ok=True)
 
    write_csv(results_folder+"datacsv/"+"data_123_raw_subject"+subject, xy123_raw)
    write_csv(results_folder+"datacsv/"+"data_123_raw_mean_subject"+subject, xy123_raw_mean)
    write_csv(results_folder+"datacsv/"+"data_123_raw_mean_CSP_subject"+subject, xy123_raw_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_123_raw_mean_CSP_LogVar_subject"+subject, xy123_raw_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_12_raw_subject"+subject, xy12_raw)
    write_csv(results_folder+"datacsv/"+"data_12_raw_mean_subject"+subject, xy12_raw_mean)
    write_csv(results_folder+"datacsv/"+"data_12_raw_mean_CSP_subject"+subject, xy12_raw_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_12_raw_mean_CSP_LogVar_subject"+subject, xy12_raw_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_13_raw_subject"+subject, xy13_raw)
    write_csv(results_folder+"datacsv/"+"data_13_raw_mean_subject"+subject, xy13_raw_mean)
    write_csv(results_folder+"datacsv/"+"data_13_raw_mean_CSP_subject"+subject, xy13_raw_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_13_raw_mean_CSP_LogVar_subject"+subject, xy13_raw_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_23_raw_subject"+subject, xy23_raw)
    write_csv(results_folder+"datacsv/"+"data_23_raw_mean_subject"+subject, xy23_raw_mean)
    write_csv(results_folder+"datacsv/"+"data_23_raw_mean_CSP_subject"+subject, xy23_raw_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_23_raw_mean_CSP_LogVar_subject"+subject, xy23_raw_mean_CSP_LogVar)    
   
    write_csv(results_folder+"datacsv/"+"data_123_morlet_subject"+subject, xy123_morlet)
    write_csv(results_folder+"datacsv/"+"data_123_morlet_mean_subject"+subject, xy123_morlet_mean)
    write_csv(results_folder+"datacsv/"+"data_123_morlet_mean_CSP_subject"+subject, xy123_morlet_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_123_morlet_mean_CSP_LogVar_subject"+subject, xy123_morlet_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_12_morlet_subject"+subject, xy12_morlet)
    write_csv(results_folder+"datacsv/"+"data_12_morlet_mean_subject"+subject, xy12_morlet_mean)
    write_csv(results_folder+"datacsv/"+"data_12_morlet_mean_CSP_subject"+subject, xy12_morlet_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_12_morlet_mean_CSP_LogVar_subject"+subject, xy12_morlet_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_13_morlet_subject"+subject, xy13_morlet)
    write_csv(results_folder+"datacsv/"+"data_13_morlet_mean_subject"+subject, xy13_morlet_mean)
    write_csv(results_folder+"datacsv/"+"data_13_morlet_mean_CSP_subject"+subject, xy13_morlet_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_13_morlet_mean_CSP_LogVar_subject"+subject, xy13_morlet_mean_CSP_LogVar)
    
    write_csv(results_folder+"datacsv/"+"data_23_morlet_subject"+subject, xy23_morlet)
    write_csv(results_folder+"datacsv/"+"data_23_morlet_mean_subject"+subject, xy23_morlet_mean)
    write_csv(results_folder+"datacsv/"+"data_23_morlet_mean_CSP_subject"+subject, xy23_morlet_mean_CSP)
    write_csv(results_folder+"datacsv/"+"data_23_morlet_mean_CSP_LogVar_subject"+subject, xy23_morlet_mean_CSP_LogVar)    
   