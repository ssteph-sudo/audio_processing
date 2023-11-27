import csv
import pandas as pd
import numpy as np
import os
from Feature_Extraction_Zero_Crossing_Rate import extract_zero_crossing
from Feature_Extraction_Spectral_Centroid import extract_spectral_centroid
from Feature_Extraction_Avg_Energy import extract_avg_energy


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

def pipeline(path, row):
    # Audio Feature 1
    avg_energy = extract_avg_energy(path)
    row.append(avg_energy)

    # Audio Feature 2
    spectral_centroid_avg = extract_spectral_centroid(path)
    row.append(spectral_centroid_avg)

    # Audio Feature 3
    zero_crossing_feature = extract_zero_crossing(path)
    row.append(zero_crossing_feature)

def generate_csv():
    data = []
    files = generate_file_list()
    headerList = ["fileName", "Avg_Energy", "Spectral_Centroid", "Zero_Crossing", "Label"]

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

        # Pass the file through the pipeline to extract features
        pipeline(path, row)

        # Label each audio file with a ground truth label
        row.append(label)

        # Add each row to the dataset
        data.append(row)

    # Save the features as a dataframe and then a csv file
    df = pd.DataFrame(data)
    print(df.shape)
    df.to_csv('features.csv', mode='w', index=False, header=headerList)

generate_csv()
