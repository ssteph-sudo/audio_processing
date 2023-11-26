import os
import tkinter as tk
from tkinter import filedialog, ttk
import pygame
import threading
import time

class AudioClassifierGUI:
    def __init__(self, root):
        self.root = root
        root.title("Audio Classifier")
        
        window_width = 600
        window_height = 400

        # Get the screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Find the center of screen
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        # Set the position of the window to the center of the screen 
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


        self.load_button = tk.Button(root, text="Load Folder", command=self.load_folder)
        self.load_button.pack(pady=10)

        self.listbox = tk.Listbox(root)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        self.play_button = tk.Button(control_frame, text="Play", command=self.play_audio)
        self.play_button.grid(row=0, column=0, padx=10)

        self.pause_button = tk.Button(control_frame, text="Pause", command=self.pause_audio)
        self.pause_button.grid(row=0, column=1, padx=10)

        self.stop_button = tk.Button(control_frame, text="Stop", command=self.stop_audio)
        self.stop_button.grid(row=0, column=2, padx=10)

        self.progress_label = tk.Label(root, text="00:00 / 00:00")
        self.progress_label.pack()

        pygame.mixer.init()
        self.file_paths = {}
        self.current_audio = None
        self.is_playing = False
        self.is_paused = False

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.populate_list(folder_path)

    def populate_list(self, folder_path):
        self.listbox.delete(0, tk.END)
        self.file_paths.clear()
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    full_path = os.path.join(root, file)
                    self.listbox.insert(tk.END, file)
                    self.file_paths[file] = full_path

    def play_audio(self):
        selected_file = self.listbox.get(tk.ACTIVE)
        if selected_file and self.file_paths.get(selected_file):
            full_path = self.file_paths[selected_file]

            if self.current_audio and self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                return

            if self.current_audio:
                self.stop_audio()
            pygame.mixer.music.load(full_path)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            threading.Thread(target=self.update_progress_label, daemon=True).start()

    def pause_audio(self):
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.current_audio = None
        self.is_playing = False
        self.is_paused = False

    def update_progress_label(self):
        while pygame.mixer.music.get_busy():
            elapsed_time = pygame.mixer.music.get_pos() // 1000
            total_time = pygame.mixer.Sound(self.file_paths[self.listbox.get(tk.ACTIVE)]).get_length()
            elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
            total_min, total_sec = divmod(int(total_time), 60)
            time_str = f"{elapsed_min:02}:{elapsed_sec:02} / {total_min:02}:{total_sec:02}"
            self.progress_label.config(text=time_str)
            time.sleep(1)

root = tk.Tk()
gui = AudioClassifierGUI(root)
root.mainloop()
