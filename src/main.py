import os

# TODO: Code for the plugin in this src file. Maybe separate ui from the backend logic. Possibly
# include init file and keep main for the main point of entry to the plugin.

import sys
import torch
import pickle
import librosa
import soundfile as sf
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QLineEdit, QHBoxLayout
from PyQt6.QtCore import Qt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_architecture import Generator

# File path to generator
current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_script_dir)

generator_path = os.path.join(parent_dir, 'models', 'generator_final.pth')
VECTORIZER_FILE = os.path.join(parent_dir, 'data', 'dataset', 'embeddings', 'tfidf_vectorizer.pkl')

# Load trained generator
generator = Generator(text_embedding_dim=256, latent_dim=126)
generator.load_state_dict(torch.load(generator_path))
generator.eval()

class SoundGeneratorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.num_pcs = 10
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel("Enter text and adjust sliders to generate sound:")
        layout.addWidget(self.label)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text here...")
        layout.addWidget(self.text_input)

        self.sliders = []
        for i in range(self.num_pcs):
            slider_layout = QHBoxLayout()
            slider_label = QLabel(f"PC {i+1}:")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-20)
            slider.setMaximum(20)
            slider.setValue(0)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)
            slider_layout.addWidget(slider_label)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)
            self.sliders.append(slider)

        self.generateButton = QPushButton("Generate Sound")
        self.generateButton.clicked.connect(self.generate_sound)
        layout.addWidget(self.generateButton)

        self.setLayout(layout)
        self.setWindowTitle("Text-to-Sound Generator")
        self.show()

    def load_vectorizer(vectorizer_path):
        """
        Load a previously saved TfidfVectorizer.
        """
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        return vectorizer

    def generate_sound(self):
        input_text = self.text_input.text()

        if not input_text:
            self.label.setText("Please enter text to generate sound.")
            return
        
        vectorizer = self.load_vectorizer(VECTORIZER_FILE)
        text_embedding = vectorizer.transform(input_text).toarray()

        #text_embedding = torch.randn(1, 256)  # Replace with real embedding

        latent_vector = torch.tensor([[s.value() / 10.0 for s in self.sliders] + [0] * (126 - len(self.sliders))])
        with torch.no_grad():
            spectrogram = generator(text_embedding, latent_vector).cpu().numpy()[0]

        waveform = librosa.feature.inverse.mel_to_audio(spectrogram)

        # Ensure waveform is a 1-dimensional float32 array
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        waveform = waveform.astype('float32')
        
        # Check for empty or invalid data
        if waveform.size == 0:
            raise ValueError("Waveform is empty.")
        if not np.isfinite(waveform).all():
            raise ValueError("Waveform contains invalid values.")

        sf.write('generated.wav', waveform, 22050)

        self.label.setText("Sound Generated! Check 'generated.wav'.")

app = QApplication(sys.argv)
ex = SoundGeneratorUI()
sys.exit(app.exec())
