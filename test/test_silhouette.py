# Write your k-means unit tests here
from cluster import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from cluster.silhouette import Silhouette
from cluster.kmeans import KMeans
import numpy as np
import pytest
import sklearn.metrics as sk

#Deepseek used as aid

def test_silhouette():
    # Perfectly Separated Clusters
    X, y = make_clusters(scale=0.5, k=3)  # Well-separated clusters
    silhouette = Silhouette()
    my_scores = silhouette.score(X, y)
    sk_scores = sk.silhouette_samples(X, y)
    assert np.allclose(my_scores, sk_scores, atol=1e-2)

    # Overlapping Clusters
    X, y = make_clusters(scale=2, k=3)  # Overlapping clusters
    my_scores = silhouette.score(X, y)
    sk_scores = sk.silhouette_samples(X, y)
    assert np.allclose(my_scores, sk_scores, atol=1e-2)

    # Single Cluster
    with pytest.raises(ValueError):
        X, y = make_clusters(scale=1, k=1)  # Single cluster
        my_scores = silhouette.score(X, y)
        pass

    # Random Labels
    X, _ = make_clusters(scale=1, k=3)
    y_random = np.random.randint(0, 3, size=X.shape[0])  # Random labels
    my_scores = silhouette.score(X, y_random)
    sk_scores = sk.silhouette_samples(X, y_random)
    assert np.allclose(my_scores, sk_scores, atol=1e-2)

    # Mismatch sizes
    with pytest.raises(ValueError):
        X, _ = make_clusters(scale=1, k=3)
        y_random = np.random.randint(0, 3, size= (X.shape[0] + 1))  # Random labels
        my_scores = silhouette.score(X, y_random)
        pass

    # Edge Case - Single Point
    with pytest.raises(ValueError):
        X = np.array([[1, 2]])
        y = np.array([0])
        my_scores = silhouette.score(X, y)
        pass

    # Test 6: Edge Case - Two Points
    with pytest.raises(ValueError):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        my_scores = silhouette.score(X, y)
        pass

def make_multipanel():
    t_clusters, t_labels = make_clusters(scale=0.4, k=3)
    km = KMeans(k=4)
    km.fit(t_clusters)
    pred_labels = km.predict(t_clusters)
    silhouette = Silhouette()
    scores = silhouette.score(t_clusters, pred_labels)

    plot_multipanel(t_clusters, t_labels, pred_labels, scores)