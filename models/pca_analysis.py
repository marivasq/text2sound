import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from model_architecture import Generator

# Get the directory of the currently running script (ie /models)
current_script_dir = os.path.dirname(__file__)
generator_file = os.path.join(current_script_dir, 'generator_final.pth')


def sample_latent_vectors(latent_vector_file):

    generator = Generator(text_embedding_dim=256, latent_dim=126)
    generator.load_state_dict(torch.load(generator_file))
    generator.eval()

    # Generate latent vectors
    num_samples = 1000  # Choose how many samples you want
    latent_dim = 100  # Replace with the actual latent space dimension used in training

    z_samples = torch.randn(num_samples, latent_dim)  # Sampling from normal distribution
    latent_vectors = z_samples.detach().cpu().numpy()

    # Save the latent vectors
    np.save(latent_vector_file, latent_vectors)

def perform_pca(latent_vector_file, pca_file):

    # Load latent vectors
    latent_vectors = np.load(latent_vector_file)

    # Perform PCA
    pca = PCA(n_components=10)  # Keep top 10 components
    principal_components = pca.fit_transform(latent_vectors)

    # Save PCA model and components
    np.save("pca_components.npy", principal_components)

def edit_latent_vector(latent_vector, pca, component_idx, strength):
    """Modify the latent vector along the given PCA component."""

    edited_vector = latent_vector + strength * pca.components_[component_idx]

    return edited_vector

def generate_modified_sound(generator, text_embedding, modified_latent_vector):

    modified_latent_vector = torch.tensor(modified_latent_vector, dtype=torch.float32).unsqueeze(0)
    generated_spectrogram = generator(text_embedding, modified_latent_vector)
    
    return generated_spectrogram



latent_vector_file = os.path.join(current_script_dir, 'latent_vectors.npy')
pca_file = os.path.join(current_script_dir, 'pca_components.npy')

sample_latent_vectors(latent_vector_file)
perform_pca(latent_vector_file, pca_file)
