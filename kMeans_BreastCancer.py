import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#load dataset
data = load_breast_cancer()
X= data.data
y= data.target

#Starndardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#apply KMeans Clustering
kmeans= KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
labels= kmeans.labels_

# Since KMeans assigns arbitrary labels (e.g., cluster 0 ≠ class 0),
# we adjust labels to match the true labels
adjusted_labels = np.where(labels == labels[0], y[0], 1 - y[0])

#Visualize using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=adjusted_labels, cmap='viridis', alpha=0.6)
plt.title("K-Means Clustering (K=2) on Breast Cancer Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
