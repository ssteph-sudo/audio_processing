import os
import tkinter as tk
from tkinter import filedialog, Listbox, messagebox
import pygame
import threading
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from feature_extraction.Feature_Extractor import generate_csv


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

        self.listbox = Listbox(root)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.test_button = tk.Button(root, text="Test Selected File", command=self.test_file)
        self.test_button.pack(pady=5)

        self.play_button = tk.Button(root, text="Play Selected File", command=self.play_audio)
        self.play_button.pack(pady=5)

        self.progress_label = tk.Label(root, text="00:00 / 00:00")
        self.progress_label.pack()

        self.model = None
        self.file_paths = {}
        self.testing_files = []

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.extract_features_and_train(folder_path)

    def extract_features_and_train(self, folder_path):
        generate_csv(folder_path)
      #  pass

    def test_file(self):
        selected_file = self.listbox.get(tk.ACTIVE)
        if selected_file and self.model:
            # Implement testing logic here. For now, it's just a placeholder.
            messagebox.showinfo("Result", f"Classification result for {selected_file}")
        else:
            messagebox.showerror("Error", "No file selected or model not trained")

    def play_audio(self):
        selected_file = self.listbox.get(tk.ACTIVE)
        if selected_file and selected_file in self.file_paths:
            file_path = self.file_paths[selected_file]
            # Play the audio file using pygame
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

root = tk.Tk()
gui = AudioClassifierGUI(root)
root.mainloop()
