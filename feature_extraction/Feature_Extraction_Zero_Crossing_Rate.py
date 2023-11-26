# The Zero-Crossing rate
# Time Domain Feature
# Speech is more variable
# Music is less variable
# https://librosa.org/doc/main/generated/librosa.feature.zero_crossing_rate.html

import librosa

def extract_zero_crossing(file_path):

    samples, sampling_rate = librosa.load(file_path)
    zero_crossings = librosa.zero_crossings(samples, pad=False)

    return sum(zero_crossings)