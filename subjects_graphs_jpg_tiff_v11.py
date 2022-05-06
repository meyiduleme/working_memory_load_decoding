#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroEngage Project
2022
"""
# =============================================================================
# This program allows to generate and save the graphs (precision curves
# and temporal generalization matrix) for all subjects. This is done from the
# .csv file containing the results of the MVPA algorithm.
#
# We display a generalization matrix by subject and by value by which
# we average the trials.
#
# The precision curves are displayed for each subject by deleting the result curves
# the same graph the curves resulting from the values for which we average.
# =============================================================================


from functions_v11 import *
from graph_functions import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from scipy.stats import wilcoxon


#______________________PARSER______________________#

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output_folder", type=str, default="./results/")
#parser.add_argument("-o", "--output_folder", type=str, default="./results_freq_")
parser.add_argument("-t", "--temporal_analysis", type=str, default="False")
parser.add_argument("-f", "--frequential_analysis", type=str, default="True")
parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["12"])
#parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["01", "02", "03", "04", "05", "06", "07", "09", "11"])
parser.add_argument("-l", "--lobes", nargs="+", type=str, default=["components_trials"])
parser.add_argument("-a", "--averaging_values",  nargs="+", type=int, default=[8])
args = parser.parse_args()
args.temporal_analysis = str_to_bool(args.temporal_analysis)
args.frequential_analysis = str_to_bool(args.frequential_analysis)


# freq choice 06 10 25 50 all
# alphaText = "0.01" alphaText = "0.05"
# evaluation_measure = "Accuracy" evaluation_measure = "AUC"
freq = "06_10_25_50"
#freq = "50"
alphaText = "0.01"
evaluation_measure = "AUC"
alpha = float(alphaText)
args.output_folder = args.output_folder+freq+"/"

# Config figure
fig, ax = plt.subplots()
plt.style.use('seaborn-whitegrid')
min_val = 0.25
curve_upper_bound = 0.99
mat_upper_bound, mat_lower_bound = 0.99 , 0.25


for lobe in args.lobes:

  if args.frequential_analysis:

    result_folder = args.output_folder + lobe + "/morlet"
    os.makedirs(result_folder+"/graphes", exist_ok=True)

    iSubject = 0
    for subject in args.list_subject:
        

        print("============================== subject: ", subject, "\n")
        
        #_______________Création des dossier de rangement_________________________#

        os.makedirs(result_folder+"/sujet_{0}/graphe/threshold_0.6".format(subject), exist_ok=True)
        os.makedirs(result_folder+"/sujet_{0}/graphe/threshold_0.55".format(subject), exist_ok=True)

        score_12, score_13, score_23 = [], [], []                  # diagonales des matrices de généralisation
        list_plus_12, list_plus_13, list_plus_23 = [], [], []           # l'erreur standard au dessus de la courbe
        list_moins_12, list_moins_13, list_moins_23 = [], [], []         # l'erreur standard en dessous de la courbe

        time = read_csv( result_folder+"/sujet_{0}/csv/time.csv".format(subject) )

        #___________________Get chance level data__________________#
        # def read_data(result_folder, sujet, averaging_value, first_class, second_class, random=False):
        rand_gen_mat_12, rand_plus_12, rand_moins_12 = read_data(result_folder, subject, 8, 1, 2, random=True)
        rand_gen_mat_13, rand_plus_13, rand_moins_13 = read_data(result_folder, subject, 8, 1, 3, random=True)
        rand_gen_mat_23, rand_plus_23, rand_moins_23 = read_data(result_folder, subject, 8, 2, 3, random=True)
        
        rand_score_12 = np.diag(rand_gen_mat_12) # A décider
        rand_score_13 = np.diag(rand_gen_mat_13) # A décider
        rand_score_23 = np.diag(rand_gen_mat_23) # A décider
        

        for i in args.averaging_values:

          gen_mat_12, plus_12, moins_12 = read_data(result_folder, subject, i, 1, 2)
          gen_mat_13, plus_13, moins_13 = read_data(result_folder, subject, i, 1, 3)
          gen_mat_23, plus_23, moins_23 = read_data(result_folder, subject, i, 2, 3)
          
          # Initialisation de la taille des matrices
          
          if subject == args.list_subject[0]:
              significative_p_12 = []
              significative_p_13 = []
              significative_p_23 = []      
                            
              subjects_significative_p_12 = np.zeros((len(gen_mat_12),len(args.list_subject)))
              subjects_significative_p_13 = np.zeros((len(gen_mat_13),len(args.list_subject)))
              subjects_significative_p_23 = np.zeros((len(gen_mat_23),len(args.list_subject)))
              
              subjects_rand_significative_p_12 = np.zeros((len(gen_mat_12),len(args.list_subject)))
              subjects_rand_significative_p_13 = np.zeros((len(gen_mat_13),len(args.list_subject)))
              subjects_rand_significative_p_23 = np.zeros((len(gen_mat_23),len(args.list_subject)))

          score_12.append(np.diag(gen_mat_12)) # A décider
          list_moins_12.append(moins_12)
          list_plus_12.append(plus_12)

          score_13.append(np.diag(gen_mat_13)) # A décider
          list_moins_13.append(moins_13)
          list_plus_13.append(plus_13)

          score_23.append(np.diag(gen_mat_23)) # A décider
          list_moins_23.append(moins_23)
          list_plus_23.append(plus_23)


        for iSample in range(len(gen_mat_12)):              
            subjects_significative_p_12[iSample][iSubject] = score_12[0][iSample]
            subjects_significative_p_13[iSample][iSubject] = score_13[0][iSample]
            subjects_significative_p_23[iSample][iSubject] = score_23[0][iSample]
            
            subjects_rand_significative_p_12[iSample][iSubject] = rand_score_12[iSample]
            subjects_rand_significative_p_13[iSample][iSubject] = rand_score_13[iSample]
            subjects_rand_significative_p_23[iSample][iSubject] = rand_score_23[iSample]
                        
        #______________Sauvegarde graphe courbe de précision_______________________#
        
        plot_acc(ax, time, evaluation_measure, score_12, list_plus_12, list_moins_12, min_val, curve_upper_bound, "subject_"+subject+
                  "_accuracy_curve_1vs2", args.averaging_values, significative_p_12, colors=["deepskyblue", "grey"],
                  chance=[rand_score_12, rand_plus_12, rand_moins_12] , smoothing_value=1 , zero=True)          
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_1vs2_freq_" + freq + ".tiff", format="tiff", dpi=300)
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_1vs2_freq_" + freq + ".jpg", format="jpg", dpi=300)
        plt.cla()
     
        plot_acc(ax, time, evaluation_measure, score_13, list_plus_13, list_moins_13, min_val, curve_upper_bound, "subject_"+subject+
                  "_accuracy_curve_1vs3", args.averaging_values, significative_p_13, colors=["deepskyblue", "grey"],
                  chance=[rand_score_13, rand_plus_13, rand_moins_13], smoothing_value=1 , zero=True)
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_1vs3_freq_" + freq + ".tiff", format="tiff", dpi=300)
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_1vs3_freq_" + freq + ".jpg", format="jpg", dpi=300)
        plt.cla()
        
        plot_acc(ax, time, evaluation_measure, score_23, list_plus_23, list_moins_23, min_val, curve_upper_bound, "subject_"+subject+
                  "_accuracy_curve_2vs3", args.averaging_values, significative_p_23, colors=["deepskyblue", "grey"],
                  chance=[rand_score_23, rand_plus_23, rand_moins_23], smoothing_value=1 , zero=True)
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_2vs3_freq_" + freq + ".tiff", format="tiff", dpi=300)
        plt.savefig(result_folder+"/graphes/subject_{0}_".format(subject) + evaluation_measure + "_curve_2vs3_freq_" + freq + ".jpg", format="jpg", dpi=300)
        plt.cla()   
        
        iSubject += 1

    
    for iSample in range(len(gen_mat_12)):              
        data1 = np.array(subjects_significative_p_12[iSample])
        data2 = np.array(subjects_rand_significative_p_12[iSample])
        stat, pvalue = wilcoxon(data1, data2)
        print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
        if pvalue > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')
            significative_p_12.append([iSample])            
            
    for iSample in range(len(gen_mat_13)):              
        data1 = np.array(subjects_significative_p_13[iSample])
        data2 = np.array(subjects_rand_significative_p_13[iSample])
        stat, pvalue = wilcoxon(data1, data2)
        print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
        if pvalue > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')
            significative_p_13.append([iSample])            

    for iSample in range(len(gen_mat_23)):              
        data1 = np.array(subjects_significative_p_23[iSample])
        data2 = np.array(subjects_rand_significative_p_23[iSample])
        stat, pvalue = wilcoxon(data1, data2)
        print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
        if pvalue > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')
            significative_p_23.append([iSample])            

    write_csv(result_folder+"/graphes/significative_12_p_"+alphaText, significative_p_12)
    write_csv(result_folder+"/graphes/significative_13_p_"+alphaText, significative_p_13)
    write_csv(result_folder+"/graphes/significative_23_p_"+alphaText, significative_p_23)    
    
    write_csv(result_folder+"/graphes/subjects_significative_12_p_"+alphaText, subjects_significative_p_12)
    write_csv(result_folder+"/graphes/subjects_significative_13_p_"+alphaText, subjects_significative_p_13)
    write_csv(result_folder+"/graphes/subjects_significative_23_p_"+alphaText, subjects_significative_p_23)
    
    
    write_csv(result_folder+"/graphes/subjects_rand_significative_12_p_"+alphaText, subjects_rand_significative_p_12)
    write_csv(result_folder+"/graphes/subjects_rand_significative_13_p_"+alphaText, subjects_rand_significative_p_13)
    write_csv(result_folder+"/graphes/subjects_rand_significative_23_p_"+alphaText, subjects_rand_significative_p_23)    

plt.close(fig)
