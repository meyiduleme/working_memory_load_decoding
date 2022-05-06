#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:07:27 2020

@author: sebastien
"""


from functions_v11 import read_csv
import numpy as np

def read_data(result_folder, sujet, averaging_value, first_class, second_class, random=False):
#def read_data(result_folder, sujet, averaging_value, second_class, random=False):

    if random:
        csv_folder = "csv_randomized_labels"
    else:
        csv_folder = "csv"

    gen_matrix = read_csv(result_folder+"/sujet_{0}/{1}/score_{2}vs{3}_av{4}.csv".format(sujet, csv_folder, first_class, second_class, averaging_value))
    plus = read_csv(result_folder+"/sujet_{0}/{1}/plus_{2}vs{3}_av{4}.csv".format(sujet, csv_folder, first_class, second_class, averaging_value))
    moins = read_csv(result_folder+"/sujet_{0}/{1}/moins_{2}vs{3}_av{4}.csv".format(sujet, csv_folder, first_class, second_class, averaging_value))

    return gen_matrix, plus, moins


def get_gen_mat_bounds(result_folder, subject, averaging_values):

    min_val = 10e3
    max_val = 0

    for i in averaging_values:

        gen_matrix_12, rand_plus_12, rand_moins_12 = read_data(result_folder, subject, i, 2)
        gen_matrix_13, rand_plus_13, rand_moins_13 = read_data(result_folder, subject, i, 3)

        if np.min(gen_matrix_12) < min_val:
            min_val = np.min(gen_matrix_12)
        if np.min(gen_matrix_13) > min_val:
            min_val = np.min(gen_matrix_13)

        if np.max(gen_matrix_12) < max_val:
            max_val = np.max(gen_matrix_12)
        if np.max(gen_matrix_13) > max_val:
            max_val = np.max(gen_matrix_13)

    bound = max( [abs(0.5 - min_val), abs(0.5 - max_val)] )

    return 0.5+bound, 0.5-bound