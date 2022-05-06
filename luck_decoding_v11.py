#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroEngage Project
2022
"""
# =============================================================================
# This program implements preprocessing and decoding steps on the
# epochs. The results are stored in a CSV file.
#
# Preprocessing: - Separation of the data by classes
# - Averaging of signals by i trials
# - CSP filter
# =============================================================================

from mne.io import read_epochs_eeglab
from mne.decoding import CSP
import numpy as np
from functions_v11 import *
from os import makedirs
import argparse


parser = argparse.ArgumentParser()

# Config Gérard
parser.add_argument("-i", "--input_data", type=str, default="./data/")
parser.add_argument("-o", "--output_folder", type=str, default="./results/")
parser.add_argument("-t", "--temporal_analysis", type=str, default="False")
parser.add_argument("-f", "--frequential_analysis", type=str, default="True")
parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["01", "02", "03", "04", "05", "06", "07", "09", "11"])
parser.add_argument("-l", "--lobes", nargs="+", type=str, default=["components_trials"])
parser.add_argument("-a", "--averaging_values",  nargs="+", type=int, default=[4])
parser.add_argument("-m", "--morlet_freqs", nargs="+", type=int, default=[6, 10, 25, 50])

# Config Sébastien
# parser.add_argument("-i", "--input_data", type=str, default="/home/sebastien/Documents/cours/ecole_des_mines/2A/IA/RD/code/last_data/")
# parser.add_argument("-o", "--output_folder", type=str, default="new_results/")
# parser.add_argument("-t", "--temporal_analysis", type=str)
# parser.add_argument("-f", "--frequential_analysis", type=str)
# parser.add_argument("-s", "--list_subject", nargs="+", type=str, default=["01", "02", "03", "04", "05", "06", "07", "09", "11"])
# parser.add_argument("-l", "--lobes", nargs="+", type=str, default=["components_trials", "components_trials_prefrontal", "components_trials_parietooccipital", "components_trials_centroparietal", "components_trials_frontocentral", "components_trials_occipital"])
# parser.add_argument("-a", "--averaging_values", nargs="+", type=int, default=[4])
# parser.add_argument("-m", "--morlet_freqs", nargs="+", type=int, default=[6, 10, 25, 50])

args = parser.parse_args()


args.temporal_analysis = str_to_bool(args.temporal_analysis)
args.frequential_analysis = str_to_bool(args.frequential_analysis)

for lobe in args.lobes:

  results_folder = args.output_folder+lobe+"/"

  for subject in args.list_subject:

    #_________________________data import______________________________#

    file = read_epochs_eeglab(input_fname = args.input_data + "Subject_" + subject+ "_Day_02_EEG_RAW_RS_BP_EP_BL_ICA_RJ_"+lobe+".set")

    dim1 = file.get_data().shape[0]
    dim2 = file.get_data().shape[1]
    dim3 = file.get_data().shape[2]

    x = file.get_data()
    y = file.events[:,2]

    x, time = cut_time(x, file)


    if args.temporal_analysis:

      #_________________calcul mat généralisation et erreurs standards_________________#

      gen_moyenne_12, gen_moyenne_13, gen_moyenne_23, plus12, plus13, plus23, moins12, moins13, moins23 = decode_temporal_data_luck(x, y, args.averaging_values, luck_decoding=True)

      #_________________sauvegarde des résultats en csv________________#

      makedirs(results_folder+"temporal/sujet_"+subject+"/csv_randomized_labels", exist_ok=True)

      write_csv(results_folder+"temporal/sujet_"+subject+"/csv_randomized_labels/time", time)
      save_csv( results_folder, subject, args.averaging_values, 
                gen_moyenne_12, gen_moyenne_13, gen_moyenne_23,
                plus12, plus13, plus23, 
                moins12, moins13, moins23, 
                luck_decoding=True )

    if args.frequential_analysis:

      x_morlet = morlet_tfr(x, file.info['sfreq'], args.morlet_freqs)

      x_morlet = np.reshape( x_morlet, (x_morlet.shape[0], x_morlet.shape[1]*x_morlet.shape[2], x_morlet.shape[3]) )
#      x_morlet = reshape_by_freq(x_morlet)

      #_________________calcul mat généralisation et erreurs standards_________________#

      gen_moyenne_12, gen_moyenne_13, gen_moyenne_23, plus12, plus13, plus23, moins12, moins13, moins23 = decode_temporal_data_luck(x_morlet, y, args.averaging_values, luck_decoding=True)

      #_________________sauvegarde des résultats en csv________________#

      makedirs(results_folder+"morlet/sujet_"+subject+
               "/csv_randomized_labels", exist_ok=True)
      write_csv(results_folder+"morlet/sujet_"+subject+
                "/csv_randomized_labels/time", time)
      save_csv(results_folder, subject, args.averaging_values,
               gen_moyenne_12, gen_moyenne_13, gen_moyenne_23,
               plus12, plus13, plus23,
               moins12, moins13, moins23,
               morlet=args.frequential_analysis, luck_decoding=True)




