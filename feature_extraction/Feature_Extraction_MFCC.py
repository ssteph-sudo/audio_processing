# https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

import librosa
import numpy as np

def extract_mfcc(file_path):
    
    samples, sampling_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=samples, sr=sampling_rate, n_mfcc=40)

    return mfccs