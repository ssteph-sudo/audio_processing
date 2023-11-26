import csv
import pandas as pd
import numpy as np
import os
from Feature_Extraction_Frequency import extract_frequency
from Feature_Extraction_MFCC import extract_mfcc
from Feature_Extraction_Zero_Crossing_Rate import extract_zero_crossing


speech_file_path = "../audio/speech/"
music_file_path = "../audio/music/"

# To prepare for machine learning process, the data may be formatted as below:
# filename, f1, f2, …, fn, label

def generate_file_list():
    
    speechFiles = os.listdir(speech_file_path)
    musicFiles = os.listdir(music_file_path)

    speechCount = int(len(speechFiles))
    musicCount = int(len(musicFiles))

    fileList = []

    for file in range(1, speechCount + 1):
        fileName = "sp" + str(file) + ".wav"
        fileList.append(fileName)

    
    for file in range(1, musicCount + 1):
        fileName = "mu" + str(file) + ".wav"
        fileList.append(fileName)

    print(len(fileList))
    return fileList

def generate_csv():
    data = []
    files = generate_file_list()
    headerList = []
    headerList.append("fileName")

    for file in range(len(files)):
        row = []
        filename = files[file]  # For all files in the file list
        row.append(filename)    # Grab the name of file

        path = ""
        label = ""

        # label has value “yes” if this file is a music clip and “no” if it is a speech file.
        if "mu" in filename:
            path = music_file_path + filename
            label = "yes"
        else:
            path = speech_file_path + filename
            label = "no"
        
        # Audio Feature 1
        # frequencies = extract_frequency(path) # Extract frequency features
        # for frequency in range(len(frequencies)):
        #     row.append(frequencies[frequency])

        # Audio Feature 2
        # mfcc_list = extract_mfcc(path)
        # for mfcc in range(len(mfcc_list)):
        #     row.append(mfcc_list[mfcc])

        # Audio Feature 3
        zero_crossing_feature = extract_zero_crossing(path)
        row.append(zero_crossing_feature)

        # Label each audio file with a ground truth label
        row.append(label)

        # Add each row to the dataset
        data.append(row)

    # Add header to the csv
    featureCount = len(data[0]) - 2

    for feature in range(1, featureCount + 1):
        headerList.append("F"+ str(feature))

    headerList.append("Label")

    # Save the features as a dataframe and then a csv file
    df = pd.DataFrame(data)
    print(df.shape)
    df.to_csv('features.csv', mode='w', index=False, header=headerList)

generate_csv()
