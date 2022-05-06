#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroEngage Project
2022
"""

import numpy as np
#from scipy.stats import sem
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.ticker import FuncFormatter


from random import randint as rdi
from random import random as rd
import csv
import os

from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from mne.time_frequency import tfr_array_morlet
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score


# =============================================================================
#
# reshape_by_freq : Prend des numpy array d'epochs auquels on a appliqué une transformée de Morlet, de dimension
#                   [trial][channel][frequence][signal_values] et réorganisent les dimensions afin d'avoir de array
#                   de dimension [frequence][trial][channel][signal_values].
#
# save_csv : Sauvegarde les matrices de généralisations et les erreurs standard dans des csv dans results_folder, il faut préciser si on a fait une transformé de morlet ou non et si on a calculer le niveau de chance ou non.
#
# decode_temporal_data : Prend l'ensemble des epochs tronqué et les labels d'un sujet ainsi que le nombre
#                        d'essais par lesquels on moyenne en paramètre. Sépare en fonction des labels, moyennes,
#                        applique un CSP, calcule les matrices de généralisation temporelle et les erreurs standards.
#
# create_training : sépare les données par classes et les met dans des listes
#
# averaging : moyenne sur n essais
#
# cut_time : tronque la fenetre temporelle de l'essai
#
# int_conf : renvoie les valeurs de l'intervalle de confiance à 95%
#
# plot_acc : plot les courbes de précision
#
# plot_acc_all_subjects : permet d'afficher l'ensemble des courbes de précision sur une même figure
#
# plot_temp_gen : affiche la matrice de generalisation temporelle
#
# plot_temp_gen_thres : affiche la matrice de generalisation temporelle avec les
#   valeurs, au dessus d'un paramètre seuil, en noir et le reste en blanc
#
# time_decod : normalise les données, entraine une régression logistique à chaque instant
#
# generalized_time_decod : normalise les données, entraine une régression
#   logistique à chaque instant et calcul de la matrice de generalisation temporel
#
# write_csv : creer un fichier csv de listes ou matrices
#
# read_csv : lit un fichier csv et renvoie son contenu en liste ou matrice
#
# shuffle_xy : melange les essais aléatoirement
#
# smooth : permet de lisser une courbe en réalisant la moyenne des a valeurs avant et après
# le point observé
#
# =============================================================================


# =============================================================================
# LUCK
# 
# =============================================================================

def decode_temporal_data_luck(epochs, labels, list_averaging_values, csp_transform = 'csp_space',
                         luck_decoding = True, njobs = -1):

    list_gen_moyenne_12 = []
    list_gen_moyenne_13 = []
    list_gen_moyenne_23 = []    
    list_plus_12 =  []
    list_plus_13 =  []
    list_plus_23 =  []
    list_moins_12 =  []
    list_moins_13 =  []
    list_moins_23 =  []

    #séparation par classes
    x1, x2, x3 = create_training_luck(epochs, labels)

    for i in list_averaging_values:

        #_________________________moyenne i par i sur chaque channel__________________________#

        x12, x23, x13, y12, y23, y13 = averaging_luck(i, x1, x2, x3, labels)

        x12, y12 = shuffle_xy(x12, y12)
        x13, y13 = shuffle_xy(x13, y13)
        x23, y23 = shuffle_xy(x23, y23)

        #__________________randomize les listes des labels______________#

        if(luck_decoding):
            y12 = randomize_labels(y12)
            y13 = randomize_labels(y13)
            y23 = randomize_labels(y23)

        #_______________________________CSP filtering_________________________#

        x12 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x12, y12)
        x13 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x13, y13)
        x23 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x23, y23)
        
        x12tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x12, y12)
        x13tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x13, y13)
        x23tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x23, y23)        

        #_______________________________Decoding________________________________#

        gen_ecart_12, gen_moyenne_12, gen_scores12 = generalized_time_decod(x12, y12, njobs)
        gen_ecart_13, gen_moyenne_13, gen_scores13 = generalized_time_decod(x13, y13, njobs)
        gen_ecart_23, gen_moyenne_23, gen_scores23 = generalized_time_decod(x23, y23, njobs)


        # TODO acc et auc

        scores12 = []
        scores13 = []
        scores23 = []

        # récupère les valeurs de la diagonale des matrices de généralisations de
        # chaque fold de la cross validation
        for j in range(len(gen_scores12)):
          scores12.append(np.diag(gen_scores12[j]))
          scores13.append(np.diag(gen_scores13[j]))
          scores23.append(np.diag(gen_scores23[j]))

        moyenne_12 = np.diag(gen_moyenne_12)
        moyenne_13 = np.diag(gen_moyenne_13)
        moyenne_23 = np.diag(gen_moyenne_23)
        
        ecart_12 = np.diag(gen_ecart_12)
        ecart_13 = np.diag(gen_ecart_13)
        ecart_23 = np.diag(gen_ecart_23)

        #_______________Calcul de l'erreur standard_______________#

        plus12, moins12 = int_conf(scores12, moyenne_12, ecart_12)
        plus13, moins13 = int_conf(scores13, moyenne_13, ecart_13)
        plus23, moins23 = int_conf(scores23, moyenne_23, ecart_23)        

        print("gen_moyenne shape : ", gen_moyenne_12.shape)
        print("plus shape : ", plus12.shape)

        list_gen_moyenne_12.append(gen_moyenne_12)
        list_gen_moyenne_13.append(gen_moyenne_13)
        list_gen_moyenne_23.append(gen_moyenne_23)
        list_plus_12.append(plus12)
        list_plus_13.append(plus13)
        list_plus_23.append(plus23)
        list_moins_12.append(moins12)
        list_moins_13.append(moins13)
        list_moins_23.append(moins23)

    return list_gen_moyenne_12, list_gen_moyenne_13, list_gen_moyenne_23, list_plus_12, list_plus_13, list_plus_23, list_moins_12, list_moins_13, list_moins_23


def create_training_luck(x, y):
    x1 = []
    x2 = []
    x3 = []

    for i in range(len(y)):
        if y[i] == 1 or y[i] == 2:
            x2.append(x[i])

        elif y[i] == 3 or y[i] == 4:
            x3.append(x[i])

        elif y[i] == 5 or y[i] == 6:
            x1.append(x[i])

#        if y[i] == 1 or y[i] == 2:
#            x3.append(x[i])
#
#        elif y[i] == 3 or y[i] == 4:
#            x1.append(x[i])
#
#        elif y[i] == 5 or y[i] == 6:
#            x2.append(x[i])

        else:
            print("Class error at index : ", i)

    return x1, x2, x3



def averaging_luck(nb_trial, data1, data2, data3, y):

    x1_mean = []
    x2_mean = []
    x3_mean = []
    y1 = []
    y2 = []
    y3 = []


    indice = [ i for i in range(3)]
    x_shuf = []
    y_shuf = []

    np.random.shuffle(indice)


    for i in range(len(data1) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data1[nb_trial * i + j])

        x1_mean.append(np.mean(list_av, axis = 0))
 #       y1.append(1)        
 #       y1.append(2) # Shuffle block
        y1.append(indice[0]+1)        

    for i in range(len(data2) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data2[nb_trial * i + j])

        x2_mean.append(np.mean(list_av, axis = 0))
#        y2.append(2)        
#        y2.append(3) # Shuffle block
        y2.append(indice[1]+1)        

    for i in range(len(data3) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data3[nb_trial * i + j])

        x3_mean.append(np.mean(list_av, axis = 0))
#        y3.append(3)
 #       y3.append(1) # Shuffle block
        y3.append(indice[2]+1)        



    #__________________Creation ndarray________________#

    x12 = np.array(x1_mean + x2_mean)
    x13 = np.array(x1_mean + x3_mean)
    x23 = np.array(x2_mean + x3_mean)

    x12 = np.ndarray(shape=(len(x12), len(x12[0]), len(x12[0][0])), buffer = x12)
    x13 = np.ndarray(shape=(len(x13), len(x13[0]), len(x13[0][0])), buffer = x13)
    x23 = np.ndarray(shape=(len(x23), len(x23[0]), len(x23[0][0])), buffer = x23)

    y12 = np.ndarray(shape=(len(y1+y2)), buffer=np.array(y1+y2), dtype=int)
    y13 = np.ndarray(shape=(len(y3+y1)), buffer=np.array(y3+y1), dtype=int)
    y23 = np.ndarray(shape=(len(y2+y3)), buffer=np.array(y2+y3), dtype=int)

    return x12, x23, x13, y12, y23, y13


# =============================================================================
# =============================================================================











def reshape_by_freq(x_morlet):

    nb_freqs = x_morlet.shape[2]
    freqs_list = []
    channel_buffer = {}
    trial_buffer = {}

    for freqs in range(nb_freqs):
        trial_buffer[ "t{0}".format(freqs) ] = []

    for trial in range(x_morlet.shape[0]):

        for freqs in range(nb_freqs):
            channel_buffer[ "c{0}".format(freqs) ] = []

        for channel in range(x_morlet.shape[1]):
            for freqs in range(nb_freqs):
                channel_buffer[ "c{0}".format(freqs) ].append( x_morlet[trial][channel][freqs] )

        for freqs in range(nb_freqs):
            trial_buffer[ "t{0}".format(freqs) ].append( channel_buffer["c{0}".format(freqs)] )

    for freqs in range(nb_freqs):
        freqs_list.append( trial_buffer["t{0}".format(freqs)] )

    return np.array(freqs_list)



def save_csv(results_folder, sujet, averaging_values,
             list_gen_moyenne_12, list_gen_moyenne_13, list_gen_moyenne_23,
             list_plus_12, list_plus_13, list_plus_23,
             list_moins_12, list_moins_13, list_moins_23,
             freq=None, morlet=False, luck_decoding=False):    

    if not morlet and not luck_decoding:
        for i in range( len(averaging_values) ):

            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/score_1vs2_av"+
                      str(averaging_values[i]), list_gen_moyenne_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/score_1vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/score_2vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_23[i])
            
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/plus_1vs2_av"+
                      str(averaging_values[i]), list_plus_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/plus_1vs3_av"+
                      str(averaging_values[i]), list_plus_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/plus_2vs3_av"+
                      str(averaging_values[i]), list_plus_23[i])            

            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/moins_1vs2_av"+
                      str(averaging_values[i]), list_moins_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/moins_1vs3_av"+
                      str(averaging_values[i]), list_moins_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv/moins_2vs3_av"+
                      str(averaging_values[i]), list_moins_23[i])            


    if luck_decoding and not morlet:
        for i in range( len(averaging_values) ):

            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/score_1vs2_av"+
                     str(averaging_values[i]), list_gen_moyenne_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/score_1vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/score_2vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_23[i])
            

            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/plus_1vs2_av"+
                      str(averaging_values[i]), list_plus_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/plus_1vs3_av"+
                      str(averaging_values[i]), list_plus_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/plus_2vs3_av"+
                      str(averaging_values[i]), list_plus_23[i])

            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/moins_1vs2_av"+
                      str(averaging_values[i]), list_moins_12[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/moins_1vs3_av"+
                      str(averaging_values[i]), list_moins_13[i])
            write_csv(results_folder + "temporal/sujet_"+sujet+"/csv_randomized_labels/moins_2vs3_av"+
                      str(averaging_values[i]), list_moins_23[i])            


    if morlet and not luck_decoding:
        for i in range( len(averaging_values) ):

            write_csv(results_folder+"morlet/sujet_" +sujet+"/csv/score_1vs2_av"+
                      str(averaging_values[i]), list_gen_moyenne_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/score_1vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/score_2vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_23[i])            

            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/plus_1vs2_av"+
                      str(averaging_values[i]), list_plus_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/plus_1vs3_av"+
                      str(averaging_values[i]), list_plus_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/plus_2vs3_av"+
                      str(averaging_values[i]), list_plus_23[i])            

            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/moins_1vs2_av"+
                      str(averaging_values[i]), list_moins_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/moins_1vs3_av"+
                      str(averaging_values[i]), list_moins_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv/moins_2vs3_av"+
                      str(averaging_values[i]), list_moins_23[i])            


    if morlet and luck_decoding:
        for i in range( len(averaging_values) ):

            write_csv(results_folder+"morlet/sujet_" +sujet+"/csv_randomized_labels/score_1vs2_av"+
                      str(averaging_values[i]), list_gen_moyenne_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/score_1vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/score_2vs3_av"+
                      str(averaging_values[i]), list_gen_moyenne_23[i])            

            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/plus_1vs2_av"+
                      str(averaging_values[i]), list_plus_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/plus_1vs3_av"+
                      str(averaging_values[i]), list_plus_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/plus_2vs3_av"+
                      str(averaging_values[i]), list_plus_23[i])            

            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/moins_1vs2_av"+
                      str(averaging_values[i]), list_moins_12[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/moins_1vs3_av"+
                      str(averaging_values[i]), list_moins_13[i])
            write_csv(results_folder + "morlet/sujet_"+sujet+"/csv_randomized_labels/moins_2vs3_av"+
                      str(averaging_values[i]), list_moins_23[i])            

    return 1


def decode_temporal_data(epochs, labels, list_averaging_values, csp_transform = 'csp_space',
                         luck_decoding = False, njobs = -1):

    list_gen_moyenne_12 = []
    list_gen_moyenne_13 = []
    list_gen_moyenne_23 = []    
    list_plus_12 =  []
    list_plus_13 =  []
    list_plus_23 =  []
    list_moins_12 =  []
    list_moins_13 =  []
    list_moins_23 =  []

    #séparation par classes
    x1, x2, x3 = create_training(epochs, labels)

    for i in list_averaging_values:

        #_________________________moyenne i par i sur chaque channel__________________________#

        x12, x23, x13, y12, y23, y13 = averaging(i, x1, x2, x3, labels)

        x12, y12 = shuffle_xy(x12, y12)
        x13, y13 = shuffle_xy(x13, y13)
        x23, y23 = shuffle_xy(x23, y23)

        #__________________randomize les listes des labels______________#

        if(luck_decoding):
            y12 = randomize_labels(y12)
            y13 = randomize_labels(y13)
            y23 = randomize_labels(y23)

        #_______________________________CSP filtering_________________________#

        x12 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x12, y12)
        x13 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x13, y13)
        x23 = CSP(n_components=6, transform_into=csp_transform).fit_transform(x23, y23)
        
        x12tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x12, y12)
        x13tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x13, y13)
        x23tmp = CSP(n_components=6, transform_into='average_power' ).fit_transform(x23, y23)        

        #_______________________________Decoding________________________________#

        gen_ecart_12, gen_moyenne_12, gen_scores12 = generalized_time_decod(x12, y12, njobs)
        gen_ecart_13, gen_moyenne_13, gen_scores13 = generalized_time_decod(x13, y13, njobs)
        gen_ecart_23, gen_moyenne_23, gen_scores23 = generalized_time_decod(x23, y23, njobs)


        # TODO acc et auc

        scores12 = []
        scores13 = []
        scores23 = []

        # récupère les valeurs de la diagonale des matrices de généralisations de
        # chaque fold de la cross validation
        for j in range(len(gen_scores12)):
          scores12.append(np.diag(gen_scores12[j]))
          scores13.append(np.diag(gen_scores13[j]))
          scores23.append(np.diag(gen_scores23[j]))

        moyenne_12 = np.diag(gen_moyenne_12)
        moyenne_13 = np.diag(gen_moyenne_13)
        moyenne_23 = np.diag(gen_moyenne_23)
        
        ecart_12 = np.diag(gen_ecart_12)
        ecart_13 = np.diag(gen_ecart_13)
        ecart_23 = np.diag(gen_ecart_23)

        #_______________Calcul de l'erreur standard_______________#

        plus12, moins12 = int_conf(scores12, moyenne_12, ecart_12)
        plus13, moins13 = int_conf(scores13, moyenne_13, ecart_13)
        plus23, moins23 = int_conf(scores23, moyenne_23, ecart_23)        

        print("gen_moyenne shape : ", gen_moyenne_12.shape)
        print("plus shape : ", plus12.shape)

        list_gen_moyenne_12.append(gen_moyenne_12)
        list_gen_moyenne_13.append(gen_moyenne_13)
        list_gen_moyenne_23.append(gen_moyenne_23)
        list_plus_12.append(plus12)
        list_plus_13.append(plus13)
        list_plus_23.append(plus23)
        list_moins_12.append(moins12)
        list_moins_13.append(moins13)
        list_moins_23.append(moins23)

    return list_gen_moyenne_12, list_gen_moyenne_13, list_gen_moyenne_23, list_plus_12, list_plus_13, list_plus_23, list_moins_12, list_moins_13, list_moins_23




















def morlet_tfr(x, sfreq, freq_list, output='power'):

    freqs =np.array(freq_list)
    n_cycles = freqs/2.

    x_morlet = tfr_array_morlet(x, sfreq = sfreq, freqs= freqs, n_cycles = n_cycles, output=output)

    return x_morlet


def create_training(x, y):
    x1 = []
    x2 = []
    x3 = []

    for i in range(len(y)):

        if y[i] == 1 or y[i] == 2:
            x2.append(x[i])

        elif y[i] == 3 or y[i] == 4:
            x3.append(x[i])

        elif y[i] == 5 or y[i] == 6:
            x1.append(x[i])

        else:
            print("Class error at index : ", i)

    return x1, x2, x3













def averaging(nb_trial, data1, data2, data3, y):


    x1_mean = []
    x2_mean = []
    x3_mean = []
    y1 = []
    y2 = []
    y3 = []

    for i in range(len(data1) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data1[nb_trial * i + j])

        x1_mean.append(np.mean(list_av, axis = 0))
        y1.append(1)

    for i in range(len(data2) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data2[nb_trial * i + j])

        x2_mean.append(np.mean(list_av, axis = 0))
        y2.append(2)

    for i in range(len(data3) // nb_trial):
        list_av = []

        for j in range(nb_trial):
            list_av.append(data3[nb_trial * i + j])

        x3_mean.append(np.mean(list_av, axis = 0))
        y3.append(3)


    #__________________Creation ndarray________________#

    x12 = np.array(x1_mean + x2_mean)
    x13 = np.array(x1_mean + x3_mean)
    x23 = np.array(x2_mean + x3_mean)

    x12 = np.ndarray(shape=(len(x12), len(x12[0]), len(x12[0][0])), buffer = x12)
    x13 = np.ndarray(shape=(len(x13), len(x13[0]), len(x13[0][0])), buffer = x13)
    x23 = np.ndarray(shape=(len(x23), len(x23[0]), len(x23[0][0])), buffer = x23)

    y12 = np.ndarray(shape=(len(y1+y2)), buffer=np.array(y1+y2), dtype=int)
    y13 = np.ndarray(shape=(len(y3+y1)), buffer=np.array(y3+y1), dtype=int)
    y23 = np.ndarray(shape=(len(y2+y3)), buffer=np.array(y2+y3), dtype=int)

    return x12, x23, x13, y12, y23, y13













def cut_time(x, file):
    t = file.times
    time_index = []
    time = []

    for i in range(len(t)):
        if -0.2 <= t[i] <= 1:
            time_index.append(i)
            time.append(t[i])

    new_x = np.ndarray(shape=(x.shape[0], x.shape[1], len(time_index)))

    for aa in range(x.shape[0]):
        for bb in range(x.shape[1]):
            for  cc in range(len(time_index)):
                new_x[aa][bb][cc] = x[aa][bb][time_index[cc]]

    return new_x, time



def int_conf(score, moyenne, ecart):
    
    delta = stats.norm.interval(0.68, loc=moyenne, scale=ecart/np.sqrt(len(score)))

    return delta[1] , delta[0]




# plot_acc(ax, time, score_12, list_plus_12, list_moins_12, min_val, curve_upper_bound, "subject_"+subject+
#                  "_accuracy_curve_1vs2", args.averaging_values, colors=["red", "grey"],
#                 chance=[rand_score_12, rand_plus_12, rand_moins_12])

def plot_acc(ax, time, evaluation_measure, score, plus, moins, bottom, top, fig_title, list_averaging, significative_p, colors, chance=False, smoothing_value=None, zero=False):

    time = [x * 1000 for x in time] # time in ms
    significative_p = [x * 1000 for x in significative_p] # significative_p in ms

    
    # score = [x * 100 for x in score] # score in %
    # plus = [x * 100 for x in plus] # plus in %
    # moins = [x * 100 for x in moins] # moins in %
    # bottom = 100 * bottom # bottom in %
    # top = 100 * top # top in %

    
    
    if smoothing_value != None:
        time = time[smoothing_value: -smoothing_value]


    #plot just 1 curve with standard error
    if(type(score[0]) == np.float64) :

        if smoothing_value != None:
            score = smooth(score, smoothing_value)
            plus = smooth(plus, smoothing_value)
            moins = smooth(moins, smoothing_value)

        ax.plot(time, score, "blue", label = "averaging "+str(list_averaging), linewidth=0.8)
        ax.fill_between(time, score, plus, facecolor="blue", alpha=0.3)
        ax.fill_between(time, score, moins, facecolor="blue", alpha=0.3)
        



    #plot several curves with standard error
    else :
        
      for i in range(len(score)):
          
          for iTime in range(len(significative_p)):
              ax.scatter(significative_p[iTime], 0.26, color = "deepskyblue", s = 0.5)
          
          if smoothing_value != None:
            score[i] = smooth(score[i], smoothing_value)
            plus[i] = smooth(plus[i], smoothing_value)
            moins[i] = smooth(moins[i], smoothing_value)

          ax.plot(time, score[i], colors[i], label = "averaging "+str(list_averaging[i]), scalex = False, scaley = False, linewidth=0.8)
          ax.fill_between(time, score[i], plus[i], facecolor=colors[i], alpha=0.3)
          ax.fill_between(time, score[i], moins[i], facecolor=colors[i], alpha=0.3)

    #plot chance level
    if chance != False:
        if smoothing_value != None:
            chance[0] = smooth(chance[0], smoothing_value)
            chance[1] = smooth(chance[1], smoothing_value)
            chance[2] = smooth(chance[2], smoothing_value)
                
        ax.plot(time, chance[0], colors[-1], label = "chance level",scalex = False, scaley = False, linewidth=0.8)
        ax.fill_between(time, chance[0], chance[1], facecolor=colors[-1], alpha=0.3)
        ax.fill_between(time, chance[0], chance[2], facecolor=colors[-1], alpha=0.3)

    #plot 0 straight lines
    # if zero:
    ax.axvline(.0, color='k', linestyle='-.', linewidth='0.5')

    # for iTime in range(len(significative_p)):
    #         # ax.scatter(200, 0.6, "blue")
    #         ax.plot(significative_p[iTime], 0.6, "blue","o")



    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time (msec)', fontsize = 9)
    ax.set_ylabel(evaluation_measure, fontsize = 9)
    ax.tick_params(labelsize = 9)
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 

    # ax.legend()
    # ax.set_title('Sensor space decoding '+str(fig_title))

    return None


def plot_acc_all_subjects(ax, time, scores, mean, bottom, top, label, fig_title, colors, smooth_value):


    new_time = time[smooth_value: -smooth_value]

    #plot all accuracy score curves
    for i in range(len(scores)):
        h1=ax.plot(new_time, smooth(scores[i], smooth_value), colors[i], label = None, scalex = False, scaley = True, linewidth=0.7)
        ax.set_ylim(bottom, top)
        
    h2=ax.plot(time, mean[0], colors[-1], label = None, scalex = False, scaley = True, linewidth=0.7)
    ax.fill_between(time, mean[0], mean[1], facecolor=colors[-1], alpha=0.3)
    ax.fill_between(time, mean[0], mean[2], facecolor=colors[-1], alpha=0.3)

    if label != None:
        labels = (label, 'chance level')
        handles = (h1[0], h2[0])
    else:
        labels = ['chance level']
        handles = [h2[0]]
    ax.legend(handles, labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    ax.set_title(fig_title)

    return None


def plot_acc_all_subjects_and_mean(ax, time, scores, mean, bottom, top, label, fig_title, colors, smooth_value):

    time = [x * 1000 for x in time] # time in ms

    new_time = time[smooth_value: -smooth_value]

    #plot all accuracy score curves
    for i in range(len(scores)):
        h1=ax.plot(new_time, smooth(scores[i], smooth_value), colors[i], label = None, scalex = False, scaley = True, linewidth=0.3)
        ax.set_ylim(bottom, top)

    h2=ax.plot(new_time, smooth(mean[0], smooth_value), colors[-1], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean[0], smooth_value), smooth(mean[1], smooth_value), facecolor=colors[-1], alpha=0.3)
    ax.fill_between(new_time, smooth(mean[0], smooth_value), smooth(mean[2], smooth_value), facecolor=colors[-1], alpha=0.3)

    if label != None:
        labels = (label, 'Mean')
        handles = (h1[0], h2[0])
    else:
        labels = ['Mean']
        handles = [h2[0]]
    ax.axvline(.0, color='k', linestyle='-.', linewidth='0.5')
    ax.legend(handles, labels)
    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time (msec)', fontsize = 6)
    ax.set_ylabel('Accuracy', fontsize = 6)
    ax.tick_params(labelsize = 6)
    # ax.set_title(fig_title)

    return None


def plot_acc_mean_new(ax, time, evaluation_measure,
                  scores12, mean12, 
                  scores13, mean13,
                  scores23, mean23,                  
                  scores_rand_12, mean_rand_12,
                  scores_rand_13, mean_rand_13,                   
                  scores_rand_23, mean_rand_23,                   
                  significative_p_12, 
                  significative_p_13,
                  significative_p_23,
                  bottom, top, label, fig_title, colors, smooth_value):

    time = [x * 1000 for x in time] # time in ms
    # significative_p_12 = [x * 1000 for x in significative_p_12] # significative_p_12 in ms
    # significative_p_13 = [x * 1000 for x in significative_p_13] # significative_p_13 in ms

    new_time = time[smooth_value: -smooth_value]
    
    significative_p = []
    for iTime in range(len(significative_p_12)):
        if (significative_p_12[iTime] >= smooth_value) & (significative_p_12[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_12[iTime]))            
    significative_p_12 = significative_p

    significative_p = []
    for iTime in range(len(significative_p_13)):
        if (significative_p_13[iTime] >= smooth_value) & (significative_p_13[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_13[iTime]))            
    significative_p_13 = significative_p    

    significative_p = []
    for iTime in range(len(significative_p_23)):
        if (significative_p_23[iTime] >= smooth_value) & (significative_p_23[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_23[iTime]))            
    significative_p_23 = significative_p    
    
    # significative_p_12 = significative_p_12[smooth_value: -smooth_value]
    # significative_p_13 = significative_p_13[smooth_value: -smooth_value]

 
    h1=ax.plot(new_time, smooth(mean12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h2=ax.plot(new_time, smooth(mean13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[2], smooth_value), facecolor=colors[1], alpha=0.3)
    
    h3=ax.plot(new_time, smooth(mean23[0], smooth_value), colors[2], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean23[0], smooth_value), smooth(mean23[1], smooth_value), facecolor=colors[2], alpha=0.3)
    ax.fill_between(new_time, smooth(mean23[0], smooth_value), smooth(mean23[2], smooth_value), facecolor=colors[2], alpha=0.3)    

    h4=ax.plot(new_time, smooth(mean_rand_12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    # ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    # ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h5=ax.plot(new_time, smooth(mean_rand_13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    # ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    # ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[2], smooth_value), facecolor=colors[1], alpha=0.3)

    h6=ax.plot(new_time, smooth(mean_rand_23[0], smooth_value), colors[2], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    # ax.fill_between(new_time, smooth(mean_rand_23[0], smooth_value), smooth(mean_rand_23[1], smooth_value), facecolor=colors[2], alpha=0.3)
    # ax.fill_between(new_time, smooth(mean_rand_23[0], smooth_value), smooth(mean_rand_23[2], smooth_value), facecolor=colors[2], alpha=0.3)

    for iTime in range(len(significative_p_12)):
        # ax.scatter(significative_p_12[iTime], 0.40, color = colors[0], s = 0.5)
        ax.scatter(new_time[int(significative_p_12[iTime])-smooth_value], 0.410, color = colors[0], s = 0.5)
        
    for iTime in range(len(significative_p_13)):
        # ax.scatter(significative_p_13[iTime], 0.41, color = colors[1], s = 0.5)
        ax.scatter(new_time[int(significative_p_13[iTime])-smooth_value], 0.405, color = colors[1], s = 0.5)
        
    for iTime in range(len(significative_p_23)):
        # ax.scatter(significative_p_13[iTime], 0.41, color = colors[2], s = 0.5)
        ax.scatter(new_time[int(significative_p_23[iTime])-smooth_value], 0.400, color = colors[2], s = 0.5)
        
    # for iTime in range(len(significative_p_12)):
    #         # ax.scatter(200, 0.6, "blue")
    #         if significative_p_12[iTime] > 5:
    #             ax.plot(time[iTime], 0.6, "blue","o")              
    # for iTime in range(len(significative_p_12)):
    #         if significative_p_12[iTime] > thresold:
    #             ax.scatter(time[iTime], 0.40, color = colors[0], s = 0.75)                
    # for iTime in range(len(significative_p_13)):
    #         if significative_p_13[iTime] > thresold:
    #             ax.scatter(time[iTime], 0.41, color = colors[1], s = 0.75)
              
    if label != None:
        labels = ('1vs2', '1vs3', '2vs3')
        handles = (h1[0], h2[0], h3[0])
    else:
        labels = ['Mean']
        handles = [h2[0]]
    ax.axvline(.0, color='k', linestyle='-.', linewidth='0.5')
    #ax.legend(handles, labels, loc='upper right')
    ax.legend(handles, labels, loc='best')

    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time (msec)', fontsize = 9)
    ax.set_ylabel(evaluation_measure, fontsize = 9)
    ax.tick_params(labelsize = 9)
    # ax.set_title(fig_title)
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 

    return None



def plot_acc_mean(ax, time,
                  scores12, mean12, scores13, mean13, 
                  scores_rand_12, mean_rand_12, scores_rand_13, mean_rand_13,                   
                  significative_p_12, significative_p_13, thresold, bottom, top, label, fig_title, colors, smooth_value):

    time = [x * 1000 for x in time] # time in ms

    new_time = time[smooth_value: -smooth_value]

 
    h1=ax.plot(new_time, smooth(mean12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h2=ax.plot(new_time, smooth(mean13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[2], smooth_value), facecolor=colors[1], alpha=0.3)

    h3=ax.plot(new_time, smooth(mean_rand_12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    # ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    # ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h4=ax.plot(new_time, smooth(mean_rand_13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    # ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    # ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[2], smooth_value), facecolor=colors[1], alpha=0.3)


    # for iTime in range(len(significative_p_12)):
    #         # ax.scatter(200, 0.6, "blue")
    #         if significative_p_12[iTime] > 5:
    #             ax.plot(time[iTime], 0.6, "blue","o")
                
    for iTime in range(len(significative_p_12)):
            if significative_p_12[iTime] > thresold:
                ax.scatter(time[iTime], 0.40, color = colors[0], s = 0.75)
                
    for iTime in range(len(significative_p_13)):
            if significative_p_13[iTime] > thresold:
                ax.scatter(time[iTime], 0.41, color = colors[1], s = 0.75)
              
    if label != None:
        labels = ('1vs2', '1vs3')
        handles = (h1[0], h2[0])
    else:
        labels = ['Mean']
        handles = [h2[0]]
    ax.axvline(.0, color='k', linestyle='-.', linewidth='0.5')
    ax.legend(handles, labels)
    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time (msec)', fontsize = 6)
    ax.set_ylabel('Accuracy', fontsize = 6)
    ax.tick_params(labelsize = 6)
    # ax.set_title(fig_title)

    return None



def plot_temp_gen(fig, ax, time, gen_matrix, bottom, top, name, smoothing=False, smooth_value=None):

    if type(gen_matrix) == list:
      new_gen_matrix = [x[:] for x in gen_matrix]
    elif type(gen_matrix) == np.ndarray:
      new_gen_matrix = [x[:].tolist() for x in gen_matrix]

    if smoothing:
      new_time = time[smooth_value: -smooth_value]
      for i in range(len(new_gen_matrix)):
        new_gen_matrix[i] = smooth(new_gen_matrix[i], smooth_value)

    if smoothing:
      im = ax.imshow(new_gen_matrix, interpolation='lanczos', origin='lower', cmap='RdBu_r',
    			   extent=(new_time[0], new_time[-1], time[0], time[-1]), vmin=bottom, vmax=top)
    if not(smoothing):
      im = ax.imshow(new_gen_matrix, interpolation='lanczos', origin='lower', cmap='RdBu_r',
    			   extent=(time[0], time[-1], time[0], time[-1]), vmin=bottom, vmax=top)

    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization '+name)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    fig.colorbar(im, ax=ax)

    return None


def plot_temp_gen_thres(ax, time, gen_matrix, thres, name, smoothing=False, smooth_value=None):

    if type(gen_matrix) == list:
      new_gen_matrix = [x[:] for x in gen_matrix]
    elif type(gen_matrix) == np.ndarray:
      new_gen_matrix = [x[:].tolist() for x in gen_matrix]

    if smoothing:
      new_time = time[smooth_value: -smooth_value]
      for i in range(len(new_gen_matrix)):
        new_gen_matrix[i] = smooth(new_gen_matrix[i], smooth_value)

    for i in range(len(new_gen_matrix)):
      for j in range(len(new_gen_matrix[0])):
        if new_gen_matrix[i][j] >= thres :
          new_gen_matrix[i][j] = 0
        else :
          new_gen_matrix[i][j] = 1

    if smoothing:
      ax.imshow(new_gen_matrix, cmap = 'gray', origin='lower', extent=(new_time[0], new_time[-1], time[0], time[-1]), vmin=0, vmax=1)
    if not(smoothing):
      ax.imshow(new_gen_matrix, cmap = 'gray', origin='lower', extent=(time[0], time[-1], time[0], time[-1]), vmin=0, vmax=1)

    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization '+name)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.legend(["thres : "+str(thres)], bbox_to_anchor=(1.35, 1))

    return None


def plot_temp_gen_thres_dual(ax, time, gen_matrix_12, gen_matrix_13, thres, name, colors, legend_labels, smoothing=False, smooth_value=None):

    if type(gen_matrix_12) == list:
      new_gen_matrix_12 = [x[:] for x in gen_matrix_12]
      new_gen_matrix_13 = [x[:] for x in gen_matrix_13]
    elif type(gen_matrix_12) == np.ndarray:
      new_gen_matrix_12 = [x[:].tolist() for x in gen_matrix_12]
      new_gen_matrix_13 = [x[:].tolist() for x in gen_matrix_13]

    if smoothing:
      new_time = time[smooth_value: -smooth_value]
      for i in range(len(new_gen_matrix_12)):
        new_gen_matrix_12[i] = smooth(new_gen_matrix_12[i], smooth_value)
        new_gen_matrix_13[i] = smooth(new_gen_matrix_13[i], smooth_value)

    new_matrix = np.zeros([len(new_gen_matrix_12), len(new_gen_matrix_12[0])])
    for i in range(len(new_gen_matrix_12)):
      for j in range(len(new_gen_matrix_12[0])):
        if new_gen_matrix_12[i][j] >= thres and new_gen_matrix_13[i][j] < thres :
          new_matrix[i][j] = 1
        if new_gen_matrix_12[i][j] < thres and new_gen_matrix_13[i][j] < thres :
          new_matrix[i][j] = 0
        if new_gen_matrix_13[i][j] >= thres and new_gen_matrix_12[i][j] < thres :
          new_matrix[i][j] = 2
        if new_gen_matrix_12[i][j] >= thres and new_gen_matrix_13[i][j] >= thres :
          new_matrix[i][j] = 3

    cmap = mpl.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if smoothing:
      img = ax.imshow(new_matrix, cmap = cmap, origin='lower', norm=norm,
                      extent=(new_time[0], new_time[-1], time[0], time[-1]), interpolation='nearest')
    if not(smoothing):
      img = ax.imshow(new_matrix, cmap = cmap, origin='lower', norm=norm,
                      extent=(time[0], time[-1], time[0], time[-1]), interpolation='nearest')

    patch1 = mpl.patches.Patch(color='blue', label=legend_labels[0])
    patch2 = mpl.patches.Patch(color='red', label=legend_labels[1])
    patch3 = mpl.patches.Patch(color='orchid', label=legend_labels[2])

    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Temporal generalization '+name)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    ax.legend(handles=[patch1,patch2,patch3], bbox_to_anchor=(1.35, 1))

    mpl.colorbar.ColorbarBase(img, cmap=cmap, norm=norm, boundaries=bounds)

    return None


def time_decod(x, y, jobs):

  clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))

  time_decod = SlidingEstimator(clf, n_jobs=jobs, scoring=make_scorer(accuracy_score), verbose=False)

  scores = cross_val_multiscore(time_decod, x, y, cv=5, n_jobs=jobs)

  score_moyen = np.mean(scores, axis=0)

  return score_moyen, scores


def generalized_time_decod(x, y, jobs):

  clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
  # time_decod = GeneralizingEstimator(clf, n_jobs=jobs, scoring=make_scorer(accuracy_score), verbose=False)   
  # scores = cross_val_multiscore(time_decod, x, y, cv=10, n_jobs=jobs)   #contient les matrices de généralisation des 10 folds
  # score_moyen = np.mean(scores, axis=0)   # calcule la matrice de généralisation moyenne sur des 10 folds
  # score_ecart = np.std(scores, axis=0)   # GD calcule la matrice de généralisation std sur des 10 folds
  time_decod = GeneralizingEstimator(clf, n_jobs=jobs, scoring=make_scorer(roc_auc_score), verbose=False)
  scores = cross_val_multiscore(time_decod, x, y, cv=10, n_jobs=jobs)   #contient les matrices de généralisation des 10 folds
  score_moyen = np.mean(scores, axis=0)   # calcule la matrice de généralisation moyenne sur des 10 folds
  score_ecart = np.std(scores, axis=0)   # GD calcule la matrice de généralisation std sur des 10 folds
    

  return score_ecart, score_moyen, scores



def write_csv(name, list_score):

  os.makedirs(os.path.dirname(name+".csv"), exist_ok=True)
  # with open(name+".csv", mode='w') as score_file:
  with open(name+".csv", mode='w', newline='') as score_file:
    score_writer = csv.writer(score_file, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    if(type(list_score) == np.float64):
        score_writer.writerow([list_score])
    elif(len(list_score)==0):
      score_writer.writerow(list_score)
    elif(type(list_score[0])==np.float64):
      score_writer.writerow(list_score)
    else:
      for i in range(len(list_score)):
        score_writer.writerow(list_score[i])
    score_file.close()

def read_csv(name):

  matrix = []

  with open(name, mode="r") as file:
    csv_reader = csv.reader(file, delimiter=",")

    if "score" in name :

      for row in csv_reader :
        matrix.append(row)

      for i in range(len(matrix)):
        for j in range(len(matrix[i])):
          matrix[i][j] = float(matrix[i][j])

    else :
      for row in csv_reader:
        for i in row :
          matrix.append(float(i))

  file.close()

  return matrix


def shuffle_xy(x, y):

  indice = [ i for i in range(len(x))]
  x_shuf = []
  y_shuf = []

  np.random.shuffle(indice)

  for i in range(len(x)):
    x_shuf.append(x[indice[i]])
    y_shuf.append(y[indice[i]])

  x_shuf = np.array(x_shuf)
  y_shuf = np.array(y_shuf)

  return x_shuf, y_shuf


def randomize_labels(labels):

  index = [i for i in range(len(labels))]
  rand_diag = []

  np.random.shuffle(index)

  for i in range(len(labels)):
    rand_diag.append(labels[index[i]])

  return np.array(rand_diag)


def smooth(scores, window):
    scores_smooth=[]

    for i in range(window, len(scores)-window):
        average = np.mean(scores[i-window : i+window+1])
        scores_smooth.append(average)

    return scores_smooth


def str_to_bool(string):

    if string == 'true' or string == 'True':
        return True
    else:
        return False




def plot_acc_mean_new_new(ax, time, evaluation_measure,
                  scores12, mean12, 
                  scores13, mean13,
                  scores23, mean23,                  
                  scores_rand_12, mean_rand_12,
                  scores_rand_13, mean_rand_13,                   
                  scores_rand_23, mean_rand_23,                   
                  significative_p_12, 
                  significative_p_13,
                  significative_p_23,
                  bottom, top, label, fig_title, colors, smooth_value):

    time = [x * 1000 for x in time] # time in ms
    # significative_p_12 = [x * 1000 for x in significative_p_12] # significative_p_12 in ms
    # significative_p_13 = [x * 1000 for x in significative_p_13] # significative_p_13 in ms

    new_time = time[smooth_value: -smooth_value]
    
    significative_p = []
    for iTime in range(len(significative_p_12)):
        if (significative_p_12[iTime] >= smooth_value) & (significative_p_12[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_12[iTime]))            
    significative_p_12 = significative_p

    significative_p = []
    for iTime in range(len(significative_p_13)):
        if (significative_p_13[iTime] >= smooth_value) & (significative_p_13[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_13[iTime]))            
    significative_p_13 = significative_p    

    significative_p = []
    for iTime in range(len(significative_p_23)):
        if (significative_p_23[iTime] >= smooth_value) & (significative_p_23[iTime] < len(time) - 2 * smooth_value):
            significative_p.append(int(significative_p_23[iTime]))            
    significative_p_23 = significative_p    
    
    # significative_p_12 = significative_p_12[smooth_value: -smooth_value]
    # significative_p_13 = significative_p_13[smooth_value: -smooth_value]

 
    h1=ax.plot(new_time, smooth(mean12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    ax.fill_between(new_time, smooth(mean12[0], smooth_value), smooth(mean12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h2=ax.plot(new_time, smooth(mean13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    ax.fill_between(new_time, smooth(mean13[0], smooth_value), smooth(mean13[2], smooth_value), facecolor=colors[1], alpha=0.3)
    
    h3=ax.plot(new_time, smooth(mean23[0], smooth_value), colors[2], label = None, scalex = False, scaley = True, linewidth=1)
    ax.fill_between(new_time, smooth(mean23[0], smooth_value), smooth(mean23[1], smooth_value), facecolor=colors[2], alpha=0.3)
    ax.fill_between(new_time, smooth(mean23[0], smooth_value), smooth(mean23[2], smooth_value), facecolor=colors[2], alpha=0.3)    

    h4=ax.plot(new_time, smooth(mean_rand_12[0], smooth_value), colors[0], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[1], smooth_value), facecolor=colors[0], alpha=0.3)
    ax.fill_between(new_time, smooth(mean_rand_12[0], smooth_value), smooth(mean_rand_12[2], smooth_value), facecolor=colors[0], alpha=0.3)
    
    h5=ax.plot(new_time, smooth(mean_rand_13[0], smooth_value), colors[1], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[1], smooth_value), facecolor=colors[1], alpha=0.3)
    ax.fill_between(new_time, smooth(mean_rand_13[0], smooth_value), smooth(mean_rand_13[2], smooth_value), facecolor=colors[1], alpha=0.3)

    h6=ax.plot(new_time, smooth(mean_rand_23[0], smooth_value), colors[2], label = None, scalex = False, scaley = True, linewidth=1, linestyle='dashed')
    ax.fill_between(new_time, smooth(mean_rand_23[0], smooth_value), smooth(mean_rand_23[1], smooth_value), facecolor=colors[2], alpha=0.3)
    ax.fill_between(new_time, smooth(mean_rand_23[0], smooth_value), smooth(mean_rand_23[2], smooth_value), facecolor=colors[2], alpha=0.3)

    for iTime in range(len(significative_p_12)):
        # ax.scatter(significative_p_12[iTime], 0.40, color = colors[0], s = 0.5)
        ax.scatter(new_time[int(significative_p_12[iTime])-smooth_value], 0.410, color = colors[0], s = 0.5)
        
    for iTime in range(len(significative_p_13)):
        # ax.scatter(significative_p_13[iTime], 0.41, color = colors[1], s = 0.5)
        ax.scatter(new_time[int(significative_p_13[iTime])-smooth_value], 0.405, color = colors[1], s = 0.5)
        
    for iTime in range(len(significative_p_23)):
        # ax.scatter(significative_p_13[iTime], 0.41, color = colors[2], s = 0.5)
        ax.scatter(new_time[int(significative_p_23[iTime])-smooth_value], 0.400, color = colors[2], s = 0.5)
        
    # for iTime in range(len(significative_p_12)):
    #         # ax.scatter(200, 0.6, "blue")
    #         if significative_p_12[iTime] > 5:
    #             ax.plot(time[iTime], 0.6, "blue","o")              
    # for iTime in range(len(significative_p_12)):
    #         if significative_p_12[iTime] > thresold:
    #             ax.scatter(time[iTime], 0.40, color = colors[0], s = 0.75)                
    # for iTime in range(len(significative_p_13)):
    #         if significative_p_13[iTime] > thresold:
    #             ax.scatter(time[iTime], 0.41, color = colors[1], s = 0.75)
              
    if label != None:
        labels = ('1vs2', '1vs3', '2vs3')
        handles = (h1[0], h2[0], h3[0])
    else:
        labels = ['Mean']
        handles = [h2[0]]
    ax.axvline(.0, color='k', linestyle='-.', linewidth='0.5')
    #ax.legend(handles, labels, loc='upper right')
    ax.legend(handles, labels, loc='best')

    ax.set_ylim(bottom, top)
    ax.set_xlabel('Time (msec)', fontsize = 9)
    ax.set_ylabel(evaluation_measure, fontsize = 9)
    ax.tick_params(labelsize = 9)
    # ax.set_title(fig_title)
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 

    return None

