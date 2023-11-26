# Average Energy
# Time Domain Feature

# Class example
# sum(amplitude1^2 + amplitude2^2 + ... amplitudeN^2) / length

import librosa

def extract_avg_energy(file_path):

    samples, sampling_rate = librosa.load(file_path)

    for sample in range(len(samples)):
        samples[sample] = samples[sample]*samples[sample]

    return sum(samples)/len(samples)


