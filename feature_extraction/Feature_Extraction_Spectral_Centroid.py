# The Spectral Centroid
# Speech has a lower spectral centroid
# Music has a higher spectral centroid
# https://librosa.org/doc/main/generated/librosa.feature.spectral_centroid.html
# https://youtu.be/j6NTatoi928

import librosa

FRAME_SIZE = 1024
HOP_LENGTH = 512

def extract_spectral_centroid(file_path):

    samples, sampling_rate = librosa.load(file_path)


    
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr = sampling_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    
    return sum(spectral_centroid)/len(spectral_centroid)