"""
Perform analysis and evaluation of dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Create output directory for plots
output_dir = os.path.join(current_dir, 'plots')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
current_dir = os.path.dirname(__file__)
csv_file = os.path.join(current_dir, 'dataset', 'processed', 'processed_metadata.csv') # Replace with your dataset file
data = pd.read_csv(csv_file, encoding='unicode_escape')

# Visualization 1: Distribution of Categories
plt.figure(figsize=(10, 6))
sns.countplot(y="category", data=data, order=data['category'].value_counts().index)
plt.title("Distribution of Categories")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(f"{output_dir}/category_distribution.png")
plt.close()  # Close the plot to free memory

# Visualization 2: Tempo Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['tempo'], bins=30, kde=True)
plt.title("Tempo Distribution")
plt.xlabel("Tempo")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_dir}/tempo_distribution.png")
plt.close()

# Visualization 3: Duration vs. Tempo Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="tempo", y="duration", data=data, hue="category", alpha=0.7)
plt.title("Duration vs. Tempo by Category")
plt.xlabel("Tempo")
plt.ylabel("Duration")
plt.tight_layout()
plt.savefig(f"{output_dir}/tempo_vs_duration.png")
plt.close()

# Add more plots as needed
print(f"Plots saved to the folder: {output_dir}")
