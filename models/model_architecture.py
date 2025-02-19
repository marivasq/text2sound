import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# TODO: Finish later. Maybe add an inference file and checkpoint folder.

# Get the directory of the currently running script (ie /models)
current_script_dir = os.path.dirname(__file__)

# Get the parent directory (ie /text2sound)
parent_dir = os.path.dirname(current_script_dir)

DATASET_DIR = os.path.join(parent_dir, 'data', 'dataset')
SPECTOGRAM_DIR = os.path.join(DATASET_DIR, 'spectograms')
EMBEDDING_DIR = os.path.join(DATASET_DIR, 'embeddings')

# Sample file to find dimension
spectogram_file = os.path.join(SPECTOGRAM_DIR, 'spec_0.npy')
embedding_file = os.path.join(EMBEDDING_DIR, 'embedding_0.npy')


# dataloader
class TextToSoundDataset(Dataset):
    def __init__(self, embedding_dir, spectrogram_dir):
        self.embedding_dir = embedding_dir
        self.spectrogram_dir = spectrogram_dir
    
    def __len__(self):
        length1 = len([f for f in os.listdir(SPECTOGRAM_DIR) if os.path.isfile(os.path.join(SPECTOGRAM_DIR, f))])
        length2 = len([f for f in os.listdir(EMBEDDING_DIR) if os.path.isfile(os.path.join(EMBEDDING_DIR, f))])
        assert length1 == length2
        return length1
    
    def __getitem__(self, idx):
        # Get metadata row
        embedding_file = os.path.join(self.embedding_dir, f"embedding_{idx}.npy")
        spectrogram_file = os.path.join(self.spectrogram_dir, f"spec_{idx}.npy")

        # Load precomputed data
        text_embedding = np.load(embedding_file)
        spectrogram = np.load(spectrogram_file)

        # Convert to PyTorch tensors
        text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        # Resize spectrogram to (128, 128)
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension (1, 1, 1025, 63)
        spectrogram = F.interpolate(spectrogram, size=(128, 128), mode="bilinear", align_corners=False)
        spectrogram = spectrogram.squeeze(0).squeeze(0)  # Remove batch and channel dimension

        if spectrogram.shape != (128, 128):
            raise ValueError(f"Unexpected spectrogram shape: {spectrogram.shape} for index {idx}")
        
        print("Text Embedding Shape:", text_embedding.shape)
        print("Spectrogram Shape:", spectrogram.shape)

        return {
            "text_embedding": text_embedding,
            "spectrogram": spectrogram
        }


class Generator(nn.Module):
    def __init__(self, text_embedding_dim, latent_dim):
        super(Generator, self).__init__()
        # Example architecture: fully connected layers followed by reshaping into spectrogram
        self.fc1 = nn.Linear(text_embedding_dim + latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 128 * 128)  # Output size as spectrogram dimensions

    def forward(self, text_embedding, z):
        # Concatenate text embedding and latent vector
        x = torch.cat((text_embedding, z), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 1, 128, 128)  # Reshape to match spectrogram dimensions
        return x


class Discriminator(nn.Module):
    def __init__(self, text_embedding_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(text_embedding_dim + 128 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, text_embedding, spectrogram):
        x = torch.cat((text_embedding, spectrogram.view(spectrogram.size(0), -1)), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
def main():

    # Initialize dataset
    dataset = TextToSoundDataset(EMBEDDING_DIR, SPECTOGRAM_DIR)

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    num_epochs = 50
    batch_size = 32
    latent_dim = 100
    save_every = 5
    lr = 0.0002

    temp = np.load(embedding_file)
    text_embedding_dim = temp.shape[0]
    print('text_embedding_dim', text_embedding_dim)

    generator = Generator(text_embedding_dim, latent_dim)
    discriminator = Discriminator(text_embedding_dim)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function (Binary Cross-Entropy for GANs)
    criterion = torch.nn.BCELoss()

    # Example training loop skeleton
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            # Load data
            text_embeddings = batch['text_embedding']  # Shape: (batch_size, text_embedding_dim)
            real_spectrograms = batch['spectrogram']   # Shape: (batch_size, 1, 128, 128)

            batch_size = text_embeddings.size(0)

            # ===========================
            # Train Discriminator
            # ===========================
            # Generate fake spectrograms
            z = torch.randn(batch_size, latent_dim)  # Random noise vector
            fake_spectrograms = generator(text_embeddings, z) # Shape: (batch_size, 1, 128, 128)

            # Create real and fake labels
            real_labels = torch.ones(batch_size, 1) # Real: 1
            fake_labels = torch.zeros(batch_size, 1) # Fake: 0
            
            # Discriminator predictions
            real_preds = discriminator(text_embeddings, real_spectrograms) # Predictions for real spectrograms
            fake_preds = discriminator(text_embeddings, fake_spectrograms.detach()) # Predictions for fake spectrograms
            
            # Calculate discriminator loss
            d_loss_real = criterion(real_preds, real_labels) # Loss for real data
            d_loss_fake = criterion(fake_preds, fake_labels) # Loss for fake data
            d_loss = d_loss_real + d_loss_fake

            # Backpropagate and update discriminator
            discriminator_optimizer.zero_grad()
            d_loss.backward()
            discriminator_optimizer.step()
            
            # ===========================
            # Train Generator
            # ===========================
            # Generate fake spectrograms and evaluate with discriminator
            g_preds = discriminator(text_embeddings, fake_spectrograms) # Predictions for fake data
            g_loss = criterion(g_preds, real_labels)  # Trick discriminator, so use real labels for fake spectrograms
            
            # Backpropagate and update generator
            generator_optimizer.zero_grad()
            g_loss.backward()
            generator_optimizer.step()

            # ===========================
            # Logging and Visualization
            # ===========================
            if batch_idx % 100 == 0:  # Log every 100 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        # TODO: add save checkpoint?

    # save trained models
    generator_file = os.path.join(current_script_dir, 'generator_final.pth')
    discriminator_file = os.path.join(current_script_dir, 'discriminator_final.pth')
    torch.save(generator.state_dict(), generator_file)
    torch.save(discriminator.state_dict(), discriminator_file)

if __name__ == '__main__':
    main()
