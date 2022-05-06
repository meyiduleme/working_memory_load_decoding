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
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
import argparse



#______________________PARSER______________________#

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output_folder", type=str, default="./results/")
#parser.add_argument("-o", "--output_folder", type=str, default="./results_freq_")
parser.add_argument("-t", "--temporal_analysis", type=str, default="False")
parser.add_argument("-f", "--frequential_analysis", type=str, default="True")
parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["12"])
#parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["01", "02", "03", "04", "05", "06", "07", "09", "11"])
parser.add_argument("-l", "--lobes", nargs="+", type=str, default=["components_trials"])
parser.add_argument("-a", "--averaging_values",  nargs="+", type=int, default=[4])
args = parser.parse_args()
args.temporal_analysis = str_to_bool(args.temporal_analysis)
args.frequential_analysis = str_to_bool(args.frequential_analysis)

# freq choice 06 10 25 50 all
# alphaText = "0.01"x²x²² alphaText = "0.05"
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
min_val = 0.39
curve_upper_bound = 0.91
mat_upper_bound, mat_lower_bound = 0.91 , 0.39
thresold = 7


for lobe in args.lobes:

    #_______contient les matrices de généralisation temporelle et randomisées______________#
    list_gen_mat_12, list_gen_mat_13, list_gen_mat_23 = [], [], []
    list_rand_gen_mat_12, list_rand_gen_mat_13, list_rand_gen_mat_23 = [], [], []
    
    #_________contient les diagonales des matrices de généralisation et géné randomisées___________#
    list_diag_gen_mat_12, list_diag_gen_mat_13, list_diag_gen_mat_23  = [], [], []
    list_rand_diag_gen_mat_12, list_rand_diag_gen_mat_13, list_rand_diag_gen_mat_23 = [], [], []

    if args.frequential_analysis:

        result_folder = args.output_folder + lobe + "/morlet"

        list_gen_mat_12.clear()
        list_gen_mat_13.clear()
        list_gen_mat_23.clear()

        list_diag_gen_mat_12.clear()
        list_diag_gen_mat_13.clear()
        list_diag_gen_mat_23.clear()

        list_rand_gen_mat_12.clear()
        list_rand_gen_mat_13.clear()
        list_rand_gen_mat_23.clear()

        list_rand_diag_gen_mat_12.clear()
        list_rand_diag_gen_mat_13.clear()
        list_rand_diag_gen_mat_23.clear()

        #__________Création de dossiers de rangement__________________#
        makedirs(result_folder+"/all_subjects", exist_ok=True)

        for subject in args.list_subject:

            time = read_csv(result_folder+"/sujet_{0}/csv/time.csv".format(subject))

            gen_mat_12 = read_csv(result_folder+"/sujet_{0}/csv/score_1vs2_av8.csv".format(subject))
            gen_mat_13 = read_csv(result_folder+"/sujet_{0}/csv/score_1vs3_av8.csv".format(subject))
            gen_mat_23 = read_csv(result_folder+"/sujet_{0}/csv/score_2vs3_av8.csv".format(subject))

            rand_gen_mat_12 = read_csv(result_folder+"/sujet_{0}/csv_randomized_labels/score_1vs2_av8.csv".format(subject))
            rand_gen_mat_13 = read_csv(result_folder+"/sujet_{0}/csv_randomized_labels/score_1vs3_av8.csv".format(subject))
            rand_gen_mat_23 = read_csv(result_folder+"/sujet_{0}/csv_randomized_labels/score_2vs3_av8.csv".format(subject))
            

            #___________ajout des matrices de généralisation et les randomizées de chaque sujet dans une liste________________#
            list_gen_mat_12.append(gen_mat_12)
            list_gen_mat_13.append(gen_mat_13)
            list_gen_mat_23.append(gen_mat_23)

            list_rand_gen_mat_12.append(rand_gen_mat_12)
            list_rand_gen_mat_13.append(rand_gen_mat_13)
            list_rand_gen_mat_23.append(rand_gen_mat_23)

            #____________ajout des diagonales des matrices de géné et géné randomisées de chaque sujet dans une liste_____________#
            list_diag_gen_mat_12.append(np.diag(gen_mat_12))
            list_diag_gen_mat_13.append(np.diag(gen_mat_13))
            list_diag_gen_mat_23.append(np.diag(gen_mat_23))

            list_rand_diag_gen_mat_12.append(np.diag(rand_gen_mat_12))
            list_rand_diag_gen_mat_13.append(np.diag(rand_gen_mat_13))            
            list_rand_diag_gen_mat_23.append(np.diag(rand_gen_mat_23))            

        #_______________calcul de la matrice de généralisation moyenne_______________#
        mean_gen_mat_12 = np.mean(list_gen_mat_12, axis = 0)
        mean_gen_mat_13 = np.mean(list_gen_mat_13, axis = 0)
        mean_gen_mat_23 = np.mean(list_gen_mat_23, axis = 0)

        mean_rand_gen_mat_12 = np.mean(list_rand_gen_mat_12, axis = 0)
        mean_rand_gen_mat_13 = np.mean(list_rand_gen_mat_13, axis = 0)
        mean_rand_gen_mat_23 = np.mean(list_rand_gen_mat_23, axis = 0)
        
        std_gen_mat_12 = np.std(list_gen_mat_12, axis = 0)
        std_gen_mat_13 = np.std(list_gen_mat_13, axis = 0)
        std_gen_mat_23 = np.std(list_gen_mat_23, axis = 0)

        std_rand_gen_mat_12 = np.std(list_rand_gen_mat_12, axis = 0)
        std_rand_gen_mat_13 = np.std(list_rand_gen_mat_13, axis = 0)
        std_rand_gen_mat_23 = np.std(list_rand_gen_mat_23, axis = 0)

        #_______________calcul de la moyenne par sujet_______________#
        mean_gen_matS_12 = np.mean(list_gen_mat_12, axis = 1)
        mean_gen_matS_12 = np.mean(mean_gen_matS_12, axis = 1)
        mean_gen_matS_13 = np.mean(list_gen_mat_13, axis = 1)   
        mean_gen_matS_13 = np.mean(mean_gen_matS_13, axis = 1)   
        mean_gen_matS_23 = np.mean(list_gen_mat_23, axis = 1)   
        mean_gen_matS_23 = np.mean(mean_gen_matS_23, axis = 1)   

        #____________calcul de la précision moyenne (diag)______________#
        mean_diag_12 = np.diag(mean_gen_mat_12)
        mean_diag_13 = np.diag(mean_gen_mat_13)
        mean_diag_23 = np.diag(mean_gen_mat_23)
        
        mean_rand_diag_12 = np.diag(mean_rand_gen_mat_12)
        mean_rand_diag_13 = np.diag(mean_rand_gen_mat_13)
        mean_rand_diag_23 = np.diag(mean_rand_gen_mat_23)
        
        std_diag_12 = np.diag(std_gen_mat_12)
        std_diag_13 = np.diag(std_gen_mat_13)
        std_diag_23 = np.diag(std_gen_mat_23)

        std_rand_diag_12 = np.diag(std_rand_gen_mat_12)
        std_rand_diag_13 = np.diag(std_rand_gen_mat_13)
        std_rand_diag_23 = np.diag(std_rand_gen_mat_23)

        #_____________calcul de l'erreur standard______________#
        mean_plus_12, mean_moins_12 = int_conf(list_diag_gen_mat_12, mean_diag_12, std_diag_12)
        mean_plus_13, mean_moins_13 = int_conf(list_diag_gen_mat_13, mean_diag_13, std_diag_13)
        mean_plus_23, mean_moins_23 = int_conf(list_diag_gen_mat_23, mean_diag_23, std_diag_23)
        
        mean_rand_plus_12, mean_rand_moins_12 = int_conf(list_rand_diag_gen_mat_12, mean_rand_diag_12, std_rand_diag_12)
        mean_rand_plus_13, mean_rand_moins_13 = int_conf(list_rand_diag_gen_mat_13, mean_rand_diag_13, std_rand_diag_13)
        mean_rand_plus_23, mean_rand_moins_23 = int_conf(list_rand_diag_gen_mat_23, mean_rand_diag_23, std_rand_diag_23)
        
        
        
        subjects_significative_p_12 = read_csv(result_folder+"/graphes/significative_12_p_"+alphaText+".csv")
        subjects_significative_p_13 = read_csv(result_folder+"/graphes/significative_13_p_"+alphaText+".csv")
        subjects_significative_p_23 = read_csv(result_folder+"/graphes/significative_23_p_"+alphaText+".csv")
        
        
        fichier = open(result_folder+"/graphes/all_" + evaluation_measure + "_av8_freq_"+freq+".txt", "w")
        
        fichier.write("Statistiques sur [-100 , +1000] msec \n")
        fichier.write("================================================ \n")
        fichier.write("Moyenne 1vs2 : " + str(np.mean(mean_diag_12)) + "\n")
        fichier.write("Ecart-type 1vs2 : " + str(np.std(mean_diag_12)) + "\n")
        fichier.write("Moyenne Rand 1vs2 : " + str(np.mean(mean_rand_diag_12)) + "\n")
        fichier.write("Ecart-type Rand 1vs2 : " + str(np.std(mean_rand_diag_12)) + "\n\n")
        
        fichier.write("Moyenne 1vs3 : " + str(np.mean(mean_diag_13)) + "\n")
        fichier.write("Ecart-type 1vs3  : " + str(np.std(mean_diag_13)) + "\n")
        fichier.write("Moyenne Rand 1vs3 : " + str(np.mean(mean_rand_diag_13)) + "\n")
        fichier.write("Ecart-type Rand 1vs3 : " + str(np.std(mean_rand_diag_13)) + "\n\n")

        fichier.write("Moyenne 2vs3 : " + str(np.mean(mean_diag_23)) + "\n")
        fichier.write("Ecart-type 2vs3  : " + str(np.std(mean_diag_23)) + "\n")
        fichier.write("Moyenne Rand 2vs3 : " + str(np.mean(mean_rand_diag_23)) + "\n")
        fichier.write("Ecart-type Rand 2vs3 : " + str(np.std(mean_rand_diag_23)) + "\n\n")        
        

        fichier.write("Statistiques sur [0 , +900] msec \n")
        fichier.write("================================================ \n")
        fichier.write("Moyenne 1vs2 : " + str(np.mean(mean_diag_12[51:281])) + "\n")
        fichier.write("Ecart-type 1vs2 : " + str(np.std(mean_diag_12[51:281])) + "\n")
        fichier.write("Moyenne Rand 1vs2 : " + str(np.mean(mean_rand_diag_12[51:281])) + "\n")
        fichier.write("Ecart-type Rand 1vs2 : " + str(np.std(mean_rand_diag_12[51:281])) + "\n\n")
        
        fichier.write("Moyenne 1vs3 : " + str(np.mean(mean_diag_13[51:281])) + "\n")
        fichier.write("Ecart-type 1vs3  : " + str(np.std(mean_diag_13[51:281])) + "\n")
        fichier.write("Moyenne Rand 1vs3 : " + str(np.mean(mean_rand_diag_13[51:281])) + "\n")
        fichier.write("Ecart-type Rand 1vs3 : " + str(np.std(mean_rand_diag_13[51:281])) + "\n\n")        

        fichier.write("Moyenne 2vs3 : " + str(np.mean(mean_diag_23[51:281])) + "\n")
        fichier.write("Ecart-type 2vs3  : " + str(np.std(mean_diag_23[51:281])) + "\n")
        fichier.write("Moyenne Rand 2vs3 : " + str(np.mean(mean_rand_diag_23[51:281])) + "\n")
        fichier.write("Ecart-type Rand 2vs3 : " + str(np.std(mean_rand_diag_23[51:281])) + "\n\n")             
        fichier.close()
        
        #________affichage des courbes de précision ________#
        colors = ["royalblue", "orange", "blueviolet"]
        plot_acc_mean_new_new(ax, time, evaluation_measure,
                      list_diag_gen_mat_12, [mean_diag_12, mean_plus_12, mean_moins_12],
                      list_diag_gen_mat_13, [mean_diag_13, mean_plus_13, mean_moins_13],
                      list_diag_gen_mat_23, [mean_diag_23, mean_plus_23, mean_moins_23],                      
                      list_rand_diag_gen_mat_12, [mean_rand_diag_12, mean_rand_plus_12, mean_rand_moins_12],
                      list_rand_diag_gen_mat_13, [mean_rand_diag_13, mean_rand_plus_13, mean_rand_moins_13],                      
                      list_rand_diag_gen_mat_23, [mean_rand_diag_23, mean_rand_plus_23, mean_rand_moins_23],                      
                      subjects_significative_p_12,
                      subjects_significative_p_13,
                      subjects_significative_p_23,                      
                      min_val, curve_upper_bound, True, "Accuracy curve 1vs2", colors, 1)
        plt.savefig(result_folder+"/graphes/all_" + evaluation_measure + "_curve_av8_freq_"+freq+"_p_"+alphaText+".tiff", format="tiff", dpi=300)
        plt.savefig(result_folder+"/graphes/all_" + evaluation_measure + "_curve_av8_freq_"+freq+"_p_"+alphaText+".jpg", format="jpg", dpi=300)
        plt.cla()


        # colors = ["red", "limegreen", "hotpink", "royalblue", "black", "orange", "blueviolet", "mediumturquoise", "sienna", "grey"]
        # colors = ["deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "deepskyblue", "grey"]
        #________affichage de toutes les courbes de précision ensemble________#
        # plot_acc_all_subjects_and_mean(ax, time, list_diag_gen_mat_12, [mean_diag_12, mean_plus_12, mean_moins_12],
        #                       min_val, curve_upper_bound, None, "Accuracy curve 1vs2", colors, 1)
        # plt.savefig(result_folder+"/graphes/all_accuracy_curve_av4_1vs2_freq_"+freq+".jpg", format="jpg", dpi=300)
        # plt.cla()

        # plot_acc_all_subjects_and_mean(ax, time, list_diag_gen_mat_13, [mean_diag_13, mean_plus_13, mean_moins_13],
        #                       min_val, curve_upper_bound, None, "Accuracy curve 1vs3", colors, 1)
        # plt.savefig(result_folder+"/graphes/all_accuracy_curve_av4_1vs3_freq_"+freq+".jpg", format="jpg", dpi=300)
        # plt.cla()
        
        
        
        
        

plt.close(fig)










