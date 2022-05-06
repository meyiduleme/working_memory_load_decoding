# NeuroEngage Project 2017 2022
# Stable decoding of working memory load through frequency bands
 Meyi Duleme, Stéphane Perrey, Gérard Dray.
 Stable decoding of working memory load through frequency bands.
 Cognitive Neuroscience, Taylor and Francis, In press, pp.1-14.
 doi: 10.1080/17588928.2022.2026312
 hal: hal-03559620
 With the invaluable help of Sébastien Marchal student at IMT Mines Alès.

This program implements a supervised learning method, Multi Variate Pattern Analysis, to decode the EEG signals of a subject and identify the instants of the signal that are discriminating with respect to the task performed by the subject.
The temporal signal can be used as well as a decomposition of Morlet frequency according to different frequencies chosen.

Dependencies:

- MNE 0.19.1
- numpy 1.17.2
- matplotlib 3.2.0
- scipy 1.4.1
- scikit-learn 0.21.3


The project is divided into several python files:

- decoding.py allows to compute the temporal generalization matrices and to save them in CSV files
- luck_decoding.py allows to compute the luck level by randomly swapping the labels of the data and then computes the temporal generalization matrices and save them in CSV files
- subjects_graphs.py allows to read CSV files, to generate and save precision curves and generalization matrices for each subject independently.
- mean_subjects_graphs.py allows to calculate the average of the generalization matrices of all subjects, to generate and save the precision curves and the image of the generalization matrix thus obtained.
- the exec.sh shell script contains the commands to run the different scripts (the order of execution is important and must be kept).

Under spyder a script can be executed by entering the following command in the console: run filename.py --argument1 value1 --argument2 value2
For the arguments that can take a list (see below) it is enough to separate each element by a space: --argument value1 value2 value3 ... 
The arguments of the different scripts are described below.


decoding_v11.py and luck_decoding_v11.py :

- input_data: (string) the path to the EEG data.
- output_folder: (string) the destination folder for the results.
- temporal_analysis : (string) enter true if you want to analyze the signal without frequency transformation and false otherwise.
- frequential_analysis: (string) enter true if you want to analyze the signal with frequency transformation and false otherwise.
- list_subject : (list) the numbers of the subjects to analyze.
- lobes : (list) variants of the data files isolate some lobes, this parameter allows to choose the data on which we work.
- averaging_values: (list) the signals of a subject are averaged over a certain number of trials, this parameter defines this number.
- morlet_freqs : (list) the frequencies to use for the Morlet transformation.


mean_subjects_graphs_jpg_tiff_v11.py and subjects_graphs_jpg_tiff_v11.py :

- output_folder: (string) the destination folder of the results.
- temporal_analysis : (string) enter true if you want to analyze the signal without frequency transformation and false otherwise.
- frequential_analysis : enter true if you want to analyze the signal with frequential transformation and false otherwise.
- list_subject : (list) the numbers of the subjects to analyze.
- lobes : (list) variants of the data files isolate some lobes, this parameter allows to choose the data on which we work.
- averaging_values: (list) the signals of a subject are averaged over a certain number of trials, this parameter defines this number.

# Example
Theta/Alpha/Beta/Gamma	 06 10 25 50

Subjects 01 and 05

run decoding_v11.py -i ./data/ -o ./results/ -t False -f True -s 01 05 -l components_trials -a 8 -m 6 10 25 50

run luck_decoding_v11.py -i ./data/ -o ./results/ -t False -f True -s 01 05 -l components_trials -a 8 -m 6 10 25 50

copy the components_trials folder on the 06_10_25_50 folder

run subjects_graphs_jpg_tiff_v11.py -o ./results/ -t False -f True -s 01 05 -l components_trials -a 8

run mean_subjects_graphs_jpg_tiff_v11.py -o ./results/ -t False -f True -s 01 05 -l components_trials -a 8

Graphs are on the folder results\components_trials\morlet\graphes