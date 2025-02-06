# Write your k-means unit tests here
from cluster import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)
from cluster.kmeans import KMeans
import numpy as np
import pytest

def test_kmeans():
    l_clusters, l_labels = make_clusters(scale=2, k=3) 
    km = KMeans(k=4)
    km.fit(l_clusters)
    kmeans_labels = km.predict(l_clusters)
    plot_clusters(l_clusters, l_labels)
    plot_clusters(l_clusters, kmeans_labels)

    assert len(set(kmeans_labels)) == 4

    assert len(km.get_centroids()) == 4

def test_kmeans_invalids():
    #invalid init
    with pytest.raises(ValueError):
        km = KMeans(k=0)
        pass

    with pytest.raises(ValueError):
        km = KMeans(k=2, tol=0)
        pass
    
    #Too many clusters for data
    with pytest.raises(ValueError):
        km = KMeans(k=5)
        l_clusters, l_labels = make_clusters(n=3) 
        km.fit(l_clusters)
        pass
    
    #Dimension mismatch
    with pytest.raises(ValueError):
        km = KMeans(k=3)
        km.fit(np.zeros((1,2,3)))
        pass

    with pytest.raises(ValueError):
        km = KMeans(k=3)
        l_clusters, l_labels = make_clusters(n=3) 
        km.fit(l_clusters)
        km.predict(np.zeros((1,2,3)))
        pass

    