import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """


    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be an ndarray")
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be an ndarray")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of labels does not match the number of observations")
        if X.shape[0] <= len(set(y)):
            raise ValueError("# labels must be < # of samples")

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        distances = cdist(X,X,metric = 'euclidean')

        for i in range(n_samples):
            #Get the cluster for this point
            cluster_label = y[i]

            #Get a mask to select all of the same cluster
            same_cluster  = (y == cluster_label)
            same_cluster[i] = False

            a_i = 0
            b_i = np.inf

            #Calculate a_i, account for single clusters
            if np.sum(same_cluster) == 0:
                a_i = 0
            else:
                a_i = np.mean(distances[i, same_cluster])

            for other_label in set(y):
                if other_label == cluster_label:
                    continue

                other_cluster = (y == other_label)
                other_i = np.mean(distances[i, other_cluster])
                if other_i < b_i:
                    b_i = other_i
            
            #Case for only one cluster
            if b_i == np.inf:
                raise ValueError("Only one cluster provided")
            
            if max(a_i, b_i) == 0:
                scores[i] = 0
            else:
                scores[i] = (b_i - a_i) / max(a_i, b_i)
            
        return scores