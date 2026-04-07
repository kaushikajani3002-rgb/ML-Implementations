import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.cluster import KMeans #(for KMeans)

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix, classification_report #(for kmedoids)

from sklearn.metrics import silhouette_score

#Load the Iris dataset
iris= datasets.load_iris()
X = iris.data
y = iris.target


k = 3 #number of clusters

#Perform K-means clustering
kmeans = KMeans(n_clusters= k, random_state=0).fit(X)

#Perform K-medoids clustering
kmedoids= KMedoids(n_clusters= k, random_state=0).fit(X)


#get cluster labels
kmeans_labels = kmeans.labels_
kmedoids_labels = kmedoids.labels_

# Compute silhouette scores
kmeans_score = silhouette_score(X, kmeans_labels)
kmedoids_score = silhouette_score(X, kmedoids_labels)

print("KMeans Silhouette Score:", kmeans_score)
print("KMedoids Silhouette Score:", kmedoids_score)

#Visualize the clusters (using first two features)
# KMeans Plot
plt.scatter(X[:, 0], X[:,1], c=kmeans_labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# KMedoids Plot
plt.scatter(X[:, 0], X[:, 1], c=kmedoids_labels, s=50, cmap='viridis')
plt.title('KMedoids Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()


