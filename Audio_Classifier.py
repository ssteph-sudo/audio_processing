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
from tkinterdnd2 import DND_FILES, TkinterDnD

class AudioClassifierGUI:
    def __init__(self, root):
        self.results = []
        self.root = root
        root.title("Audio Classifier by S. Murray and B. Powell")
        window_width = 580
        window_height = 650

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculate x and y coordinates for the Tk root window
        x = (screen_width/2) - (window_width/2)
        y = (screen_height/2) - (window_height/2)

        # Set the dimensions of the screen place it in middle
        root.geometry('%dx%d+%d+%d' % (window_width, window_height, x, y))

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # Create frames
        top_frame = tk.Frame(root)
        bottom_frame = tk.Frame(root)

        top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Top Frame Components
        self.train_label = tk.Label(top_frame, text="Training Files")
        self.train_label.pack(side=tk.LEFT, padx=10)
        self.train_listbox = Listbox(top_frame, bg='light gray', width=50, height=20)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.Y)
        self.train_listbox.bind("<Button-1>", lambda event: "break")
        top_button_frame = tk.Frame(top_frame)
        top_button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.load_button = tk.Button(top_button_frame, text="Load Folder of .wav files", command=self.load_folder)
        self.load_button.pack()

        # Bottom Frame Components
        self.test_label = tk.Label(bottom_frame, text="Testing Files")
        self.test_label.pack(side=tk.LEFT, padx=10)
        self.test_listbox = Listbox(bottom_frame, width=50, height=15)
        self.test_listbox.pack(side=tk.LEFT, fill=tk.Y)
        bottom_button_frame = tk.Frame(bottom_frame)
        bottom_button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.test_button = tk.Button(bottom_button_frame, text="Test Selected File", command=self.test_file)
        self.test_button.pack(pady=5)
        self.play_button = tk.Button(bottom_button_frame, text="Play Selected File", command=self.play_audio)
        self.play_button.pack(pady=5)

        # Attributes
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
        #self.root.after(2000, lambda: messagebox.showinfo("Loading Complete", "Files loaded successfully!"))

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
                print("Selected Row:")
                print(selected_row)

                if not selected_row.empty:
                    # Extract the features for the selected file
                    features = selected_row.drop(["Label", "fileName"], axis=1)
                    normalized_features = scaler.transform(features)

                    # Debugging message to check the loaded features
                    #print("Loaded Features:")
                    #print(features)
                   # print("Columns:")
                    #print(features_data.columns)

                    #print("Normalized Features:")
                    #print(normalized_features)

                    # Predict whether the audio file contains speech or music
                    prediction = self.model.predict(normalized_features)

                    # Debugging message to check the prediction
                    #print("Prediction:")
                    #print(prediction)
                    self.results.append((selected_file, prediction[0], ground_truth))

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
    
    def print_summary(self):
        print("File, Model Output, Ground Truth Label")
        for file_name, model_output, ground_truth in self.results:
            print(f"{file_name}, Model output: {model_output}, Ground truth label: {ground_truth}")

root = tk.Tk()
gui = AudioClassifierGUI(root)
root.mainloop()
gui.print_summary()
