# The Spectral Centroid
# Speech has a lower spectral centroid
# Music has a higher spectral centroid
# https://librosa.org/doc/main/generated/librosa.feature.spectral_centroid.html
# https://youtu.be/j6NTatoi928

# numerator = (amp1 * freq.1) + (amp2 *freq.2) + (ampN *freq.N)
# denominator = amp1 + amp2 + amp3 + … ampN

import librosa
import numpy as np
from numpy.fft import rfft

FRAME_SIZE = 1024
HOP_LENGTH = 512

def extract_spectral_centroid(file_path):

    samples, sampling_rate = librosa.load(file_path)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr = sampling_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    
    return sum(spectral_centroid)/len(spectral_centroid)


