import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        
        if not isinstance(tol, float) or tol <= 0:
            raise ValueError("tol must be a positive integer.")
        
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        
        self.k = k
        self.tolerance = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = float("inf")
        self.labels = None
        
        np.random.seed(32)

    def _initialize_centroids(self, mat: np.ndarray):
        n_samples = mat.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = mat[random_indices]     

    def _assign_labels(self, mat:np.ndarray):
        distances = cdist(mat, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1), distances

    def _update_centroids(self, mat:np.ndarray, labels: np.ndarray):
        new_centroids = np.zeros((self.k, mat.shape[1]))
        for i in range(self.k):
            cluster_points = mat[labels == i]
            #If a cluster has no assigned points, reinitialize to new random point
            if len(cluster_points) == 0:
                new_centroids[i] = mat[np.random.randint(mat.shape[0])]
            else:
                new_centroids[i] = cluster_points.mean(axis=0)
        return new_centroids

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be a 2-D Numpy Array")
        elif mat.shape[0] < self.k:
            raise ValueError(f"Cannot apply {self.k} clusters to less then {self.k} points")
        
        self._initialize_centroids(mat)
        
        for i in range(self.max_iter):
            self.labels, distances = self._assign_labels(mat)
            new_centroids = self._update_centroids(mat, self.labels)

            new_error = np.sum(np.min(distances, axis=1) ** 2) / mat.shape[0]
            if np.abs(self.error - new_error) < self.tolerance:
                break

            self.centroids = new_centroids
            self.error = new_error
    
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise ValueError("Must fit the model before you predict")
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be a 2-D Numpy Array")
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Input data must be same number of features as fitted data")
        
        labels, distances = self._assign_labels(mat)
        return labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.error is None:
            raise ValueError("Model has not been fit yet")
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model has not been fit yet")
        return self.centroids
