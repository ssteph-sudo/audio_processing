import os
import tkinter as tk
from tkinter import filedialog, Listbox, messagebox
import pygame
import threading
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from feature_extraction.Feature_Extractor import generate_csv
from classifier.classifier import train_model 
import joblib

class AudioClassifierGUI:
    def __init__(self, root):
        self.root = root
        root.title("Audio Classifier by S. Murray and B. Powell")
        root.geometry('800x600')  # Window size

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # GUI Components
        self.load_button = tk.Button(root, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=10)

        self.train_label = tk.Label(root, text="The Files Were Slected as Training Files (0)")
        self.train_label.pack()


        self.train_listbox = Listbox(root, bg='light gray') 
        self.train_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Disable selection in training files listbox
        self.train_listbox.bind("<Button-1>", lambda event: "break")


        self.test_label = tk.Label(root, text="Click on a Testing File Below to Play or Test (0)")
        self.test_label.pack()

        self.test_listbox = Listbox(root)
        self.test_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        self.test_button = tk.Button(root, text="Test Selected File", command=self.test_file)
        self.test_button.pack(pady=5)

        self.play_button = tk.Button(root, text="Play Selected File", command=self.play_audio)
        self.play_button.pack(pady=5)

        self.progress_label = tk.Label(root, text="00:00 / 00:00")
        self.progress_label.pack()

        self.model = None
        self.file_paths = {}
        self.train_files = []
        self.test_files = []

    def load_folder(self):
        
        folder_path = filedialog.askdirectory()
        messagebox.showinfo("Please Wait", "Loading files. Please wait a few seconds...")
        if folder_path:
            # Populate file_paths dictionary
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    self.file_paths[file] = full_path
            self.extract_features_and_train(folder_path)
        self.root.after(2000, lambda: messagebox.showinfo("Loading Complete", "Files loaded successfully!"))

    def extract_features_and_train(self, folder_path):
        generate_csv(folder_path)
        x_train, x_test, y_train, y_test, self.model, self.train_files, self.test_files, self.test_labels = train_model()
        # Save the model for later use
        joblib.dump(self.model, 'audio_classifier_model.pkl')

        self.update_file_lists()

    def update_file_lists(self):
          # Clear existing lists
        self.train_listbox.delete(0, tk.END)
        self.test_listbox.delete(0, tk.END)

        # Populate training files list (non-clickable)
        for file in self.train_files:
            self.train_listbox.insert(tk.END, file)

        # Populate testing files list (clickable)
        for file in self.test_files:
            self.test_listbox.insert(tk.END, file)



    def test_file(self):
        selected_file = self.test_listbox.get(tk.ACTIVE)
        if selected_file:
            # Load the model for testing
            self.model = joblib.load('audio_classifier_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            if selected_file in self.test_labels:
                # Get the ground truth label from the dictionary
                ground_truth = "Music" if self.test_labels[selected_file] == "yes" else "Speech"

                # Load the features from the CSV file
                features_data = pd.read_csv('feature_extraction/features.csv')

                # Find the row corresponding to the selected file
                selected_row = features_data[features_data['fileName'] == selected_file]

                if not selected_row.empty:
                    # Extract the features for the selected file
                    features = selected_row.drop(["Label", "fileName"], axis=1)
                    normalized_features = scaler.transform(features)
                    """ dummy_data = {
                    "Label": ["yes"],  
                    "fileName": ["mu1.wav"], 
                    "Avg_Energy": [0.013387602993281478],  
                    "Spectral_Centroid": [7.236733455814755],
                    "Zero_Crossing": [0.09323464117433852],
                     }
                    
                    dummy_row = pd.DataFrame(dummy_data)

                    features = dummy_row.drop(["Label", "fileName"], axis=1)"""

                    # Debugging message to check the loaded features
                    print("Loaded Features:")
                    print(features)
                    print("Normalized Features:")
                    print(normalized_features)

                    # Predict whether the audio file contains speech or music
                    prediction = self.model.predict(normalized_features)

                    # Debugging message to check the prediction
                    print("Prediction:")
                    print(prediction)

                    # Convert the prediction to a readable label
                    predicted_label = "Music" if prediction[0] == 1 else "Speech"

                    # Display the results
                    messagebox.showinfo("Result", f"File: {selected_file}\nGround Truth: {ground_truth}\nPrediction: {predicted_label}")
                else:
                    messagebox.showerror("Error", "Selected file not found in the feature data")
            else:
                messagebox.showerror("Error", "Ground truth label not found for selected file")
        else:
            messagebox.showerror("Error", "No file selected")

    def play_audio(self):
        selected_file = self.test_listbox.get(tk.ACTIVE) 
        if selected_file and selected_file in self.file_paths:
            file_path = self.file_paths[selected_file]
            # Play the audio file using pygame
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

root = tk.Tk()
gui = AudioClassifierGUI(root)
root.mainloop()
