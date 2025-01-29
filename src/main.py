import os

# TODO: Code for the plugin in this src file. Maybe separate ui from the backend logic. Possibly
# include init file and keep main for the main point of entry to the plugin.

import sys
import torch
import librosa
import soundfile as sf
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton
from PyQt6.QtCore import Qt

from model_architecture import Generator  # Import your model

# Load trained generator
generator = Generator(text_embedding_dim=512, latent_dim=128)
generator.load_state_dict(torch.load("final_generator.pth"))
generator.eval()

class SoundGeneratorUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Move sliders to generate sound!")
        layout.addWidget(self.label)

        self.sliders = []
        for _ in range(3):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-20)
            slider.setMaximum(20)
            slider.setValue(0)
            slider.setTickInterval(1)
            layout.addWidget(slider)
            self.sliders.append(slider)

        self.generateButton = QPushButton("Generate Sound")
        self.generateButton.clicked.connect(self.generate_sound)
        layout.addWidget(self.generateButton)

        self.setLayout(layout)
        self.setWindowTitle("Text-to-Sound Generator")
        self.show()

    def generate_sound(self):
        text_embedding = torch.randn(1, 512)  # Replace with real embedding

        latent_vector = torch.tensor([[s.value() / 10.0 for s in self.sliders] + [0] * (128 - 3)])
        spectrogram = generator(text_embedding, latent_vector).detach().numpy()[0]

        waveform = librosa.feature.inverse.mel_to_audio(spectrogram)
        sf.write("generated.wav", waveform, 22050)

        self.label.setText("Sound Generated! Check 'generated.wav'.")

app = QApplication(sys.argv)
ex = SoundGeneratorUI()
sys.exit(app.exec())
