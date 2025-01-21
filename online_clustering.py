import tqdm
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import numpy as np
import pickle

class OnlineClustering:
    def __init__(self, distance_threshold=500):
        """
        Initialize the online clustering algorithm.

        :param distance_threshold: Maximum Euclidean distance for assigning to an existing cluster.
        """
        self.distance_threshold = distance_threshold
        self.clusters = []  # List of cluster centroids (1D arrays)
        self.cluster_assignments = []  # List to track which cluster each image belongs to

    def _flatten_image(self, image):
        """
        Flatten a 64x64x3 image into a 1D vector.

        :param image: Input image of shape (64, 64, 3).
        :return: Flattened image as a 1D vector.
        """
        return image.flatten()

    def process_image(self, image):
        """
        Process a new image and assign it to a cluster.

        :param image: Input image of shape (64, 64, 3).
        :return: The index of the cluster the image is assigned to.
        """
        image_vector = self._flatten_image(image)

        # Compute distances to existing cluster centroids
        if not self.clusters:  # If no clusters exist, create the first one
            self.clusters.append(image_vector)
            self.cluster_assignments.append(0)
            return 0

        # Ensure all centroids are 1D before calculating distances
        distances = [euclidean(image_vector, np.asarray(centroid)) for centroid in self.clusters]

        # Check if the image fits into an existing cluster
        min_distance = min(distances)
        if min_distance <= self.distance_threshold:
            cluster_index = distances.index(min_distance)
        else:
            # Create a new cluster
            cluster_index = len(self.clusters)
            self.clusters.append(image_vector)
            print(f'cluster {cluster_index} created')

        self.cluster_assignments.append(cluster_index)
        return cluster_index


# Load the saved data
data = np.load("data/small_env_5_5_3actions_100k_low_orient.npz")

# Extract the 'image' key from the data
# Assuming the shape of 'image' is (steps, height, width, channels)
images = data["image"]


# Initialize the clustering model
clustering = OnlineClustering(distance_threshold=500 * 30)

# Run clustering on the images
cluster_indices = []
for i in tqdm.tqdm(images, desc='Online Clustering Process'):
    cluster_index = clustering.process_image(i)
    cluster_indices.append(cluster_index)

# Save the clustering model and results
output = {
    'cluster_indices': cluster_indices,
    'clustering_model': clustering
}

with open('clustering_result.pkl', 'wb') as f:
    pickle.dump(output, f)

print("Clustering model and results saved to 'clustering_result.pkl'.")
