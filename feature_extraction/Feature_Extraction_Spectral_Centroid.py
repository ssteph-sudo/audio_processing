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
    # frequencies = np.fft.fft(samples)

    # # Converts the frequencies to real numbers
    # frequency_magnitudes = np.abs(frequencies)

    # # Figure out the numerator and denominator for the Spectral Centroid formula
    # numerator = 0
    # denominator = 0

    # for amplitude, frequency in enumerate(frequency_magnitudes):
    #     numerator += amplitude * frequency
    #     denominator += amplitude

    # print("samples")
    # print(samples.shape[0])
    # print(samples[0:5])
    # print("==============")

    # print("frequency magnitudes")
    # print(frequency_magnitudes.shape[0])
    # print(frequency_magnitudes[0:5])
    # print("==============")

    # https://gist.github.com/endolith/359724/aa7fcc043776f16f126a0ccd12b599499509c3cc
    spectrum = abs(rfft(samples))
    normalized_spectrum = spectrum / sum(spectrum)  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum)

    #return numerator/denominator
    return spectral_centroid

print(extract_spectral_centroid("../audio/music/mu1.wav"))