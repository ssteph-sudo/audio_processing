import os
import pandas as pd

from .Feature_Extraction_Zero_Crossing_Rate import extract_zero_crossing
from .Feature_Extraction_Spectral_Centroid import extract_spectral_centroid
from .Feature_Extraction_Avg_Energy import extract_avg_energy
from .Feature_MFCCS import extract_mfcc

def generate_file_list(dir_path):
    print(f"Generating file list from directory: {dir_path}")
    file_list = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                file_list.append(full_path)  # Append the full path
    print(f"Found {len(file_list)} files.")
    return file_list

def pipeline(path, row):
    print(f"Extracting features from {path}")
    try:
        avg_energy = extract_avg_energy(path)
        row.append(avg_energy)

        spectral_centroid_avg = extract_spectral_centroid(path)
        row.append(spectral_centroid_avg)

        zero_crossing_feature = extract_zero_crossing(path)
        row.append(zero_crossing_feature)

        mfcc_features = extract_mfcc(path)
        row.extend(mfcc_features)  # Use extend instead of append
    except Exception as e:
        print(f"Error during feature extraction from {path}: {e}")

def determine_label(filename):
    return "yes" if "mu" in filename else "no"

def generate_csv(dir_path):
    print(f"Path received in generate_csv: {dir_path}")
    data = []
    files = generate_file_list(dir_path)
    headerList = ["fileName", "Avg_Energy", "Spectral_Centroid", "Zero_Crossing"]
    headerList.extend([f"MFCC_{i+1}" for i in range(13)])  # Add headers for each MFCC feature
    headerList.append("Label")

    for path in files:
        row = [os.path.basename(path)]  # Use only the filename

        pipeline(path, row)

        label = determine_label(path)  # Determine label after feature extraction
        row.append(label)

        data.append(row)

    df = pd.DataFrame(data, columns=headerList)
    print(f"DataFrame shape: {df.shape}")
    if df.empty:
        print("Warning: The DataFrame is empty. No data extracted.")
    else:
        print(df.head())  # Print the first few rows of the DataFrame

    output_path = 'feature_extraction/features.csv'
    df.to_csv(output_path, mode='w', index=False)
    print(f"Features saved to CSV file at: {output_path}")
