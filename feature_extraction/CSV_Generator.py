import csv
import pandas as pd
import numpy as np
import os
from Feature_Extraction_Frequency import extract_frequency


speech_file_path = "../audio/speech/"
music_file_path = "../audio/music/"

# To prepare for machine learning process, the data may be formatted as below:
# filename, f1, f2, …, fn, label

def generate_file_list():
    
    speechFiles = os.listdir(speech_file_path)
    musicFiles = os.listdir(music_file_path)

    # Use only about 60 percent of the data for training data
    speechCount = int(len(speechFiles)*2/3)
    musicCount = int(len(musicFiles)*2/3)

    fileList = []

    for file in range(1, speechCount + 1):
        fileName = "sp" + str(file) + ".wav"
        fileList.append(fileName)

    
    for file in range(1, musicCount + 1):
        fileName = "mu" + str(file) + ".wav"
        fileList.append(fileName)

    return fileList

def generate_csv():
    data = []
    files = generate_file_list()

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
        frequencies = extract_frequency(path) # Extract frequency features
        for frequency in range(len(frequencies)):
            row.append(frequencies[frequency])

        # Audio Feature 2
        # Audio Feature 3
        row.append(label) # Add the label for that file

        data.append(row)

    # using the savetxt
    # from the numpy module
    df = pd.DataFrame(data)
    print(df.shape)
    df.to_csv('features.csv', mode='w', index=False, header=False)

generate_csv()
