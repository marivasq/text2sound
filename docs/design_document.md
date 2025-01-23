# Design Document - Text2Sound

## Overview

Develop a DAW plugin that generates editable sound effects from textual descriptions using AI.

## Research Background

### State of the Art Review:

- **AudioCraft by Meta**  
  AudioGen employs a transformer-based generative framework, pre-trained on public datasets of sound effects.

- **AudioLDM**  
  By using latent diffusion, AudioLDM achieves superior quality while reducing computational overhead.

- **Retrieval Augmented AudioLDM**  
  Combines pre-trained text and audio encoders with retrieval-based data augmentation.

- **GanSpace**  
  Describes a simple technique to analyze General Adversarial Networks (GANs) and create interpretable controls for image synthesis.

## Functional Specifications

### User Features:
- **Text Prompt Input**
- **Real-time sound effect playback**
- **Save and export options for audio files**
- **Editable parameters based on PCA**

### Integration:
- **Compatibility with DAWs** like Ableton and Reaper

## Technical Specifications

### Model Architecture:
- **GAN-based model**  
  _Reasoning:_ Though many state-of-the-art models use diffusion or transformer-based architectures, diffusion models can be more compute-heavy with considerably slower inference times, and transformers would not lend themselves to PCA editable parameters as they do not operate in the latent space.

### GAN Options:
- **WaveGAN**  
  A GAN architecture designed specifically for audio waveform synthesis.
  
- **StyleGAN**  
  Known for its latent space manipulation, StyleGAN can be adapted to work with audio features instead of images.
  
- **GAN-TTS**  
  A GAN for speech synthesis, which might inspire non-speech audio synthesis.
  
- **GANSynth**  
  Developed by Google Magenta, it generates high-quality audio samples in a musical context.

### Conclusion:
- **SpecGAN**  
  _Justification:_ Spectrogram-based GANs align well with text embeddings and allow for detailed sound editing in the frequency domain.

## Implementation Plan

### Phases:
1. **Research and Data Collection**
   - Literature Review 🗹
   - Dataset Sourcing 🗹
   - Data Preprocessing 🗹
   - Define Evaluation Metrics
2. **Model Training and Fine Tuning**
   - GAN Architecture Selection 
   - Model Implementation
   - Training Setup
   - Fine-tuning
   - Checkpoint Management
3. **Plugin Development**
   - Framework Setup
   - Frontend Development
   - Backend Integration
   - Audio Rendering
   - Real-Time Optimizations
4. **Testing and Debugging**
   - Functional Testing
   - Performance Testing
   - User Testing
   - Debugging

## Dataset & Training

- **Dataset Sources:** Freesound, Other options: AudioCaps, AudioSet, or custom dataset
- **Preprocessing:** Normalize audio (and Tokenize text prompts?)
- **Evaluation Metrics:** Frechet Audio Distance (FAD), cosine similarity

## Testing & Quality Assurance

### Testing Strategy:
- Unit testing for plugin functionality
- User testing for ease of use

### Performance Benchmarks:
- Latency in real-time usage
- Accuracy of generated sound effects
