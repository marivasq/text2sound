import numpy as np
from sklearn.decomposition import PCA

# Load latent vectors
latent_vectors = np.load("latent_vectors.npy")

# Perform PCA
pca = PCA(n_components=10)  # Keep top 10 components
principal_components = pca.fit_transform(latent_vectors)

# Save PCA model and components
np.save("pca_components.npy", principal_components)

# TODO: fix this file