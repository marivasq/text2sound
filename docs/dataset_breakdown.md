# Dataset Breakdown

## Overview

All relevant files are contained within the /data folder.

## Scripts

download_sounds.py - File to download sounds from Freesound using their API.
preprocess_dataset.py - File to process the data before model training.

## Data

Strategy:
Start with a small dataset of 500 samples for initial testing. Scale up to 5,000 samples.

### Categories

- kick
- snare
- hihat
- clap
- tom
- cymbal
- shaker
- bass
- other?

dataset - Houses different categories of percussive sounds.
dataset/raw - Data before preprocessing.
dataset/processed - Data after preprocessing.

### Metadata.csv

- filename
- sound id
- category
- tempo
- tags
- description
- num_downloads
- duration
- license
- username

dataset/raw/raw_metadata.csv - Metadata before preprocessing.
dataset/raw/raw_metadata.csv - Metadata after preprocessing.