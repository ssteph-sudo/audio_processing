import librosa
import numpy as np

def extract_frequency(file_path):
    samples, sampling_rate = librosa.load(file_path)
    fourier = np.fft.fft(samples)
    magnitude_spectrum = np.abs(fourier)
    frequency = np.linspace(0, sampling_rate, len(magnitude_spectrum))

    return frequency

