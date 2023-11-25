# The Zero-Crossing rate
import librosa

def extract_zero_crossing(file_path):
    samples, sampling_rate = librosa.load(file_path)
    zero_crossings = librosa.zero_crossings(samples, pad=False)
    return sum(zero_crossings)