import numpy as np

class PCA:

    def __init__(self, n_compoents):
        self.n_compoents = n_compoents
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering, subtract the mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute cov(X,X), transpose because functions need samples as columns
        cov = np.cov(X.T)

        # compute eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # for easier computation, transpose eigenvectors, because eigenvectors = [:, i] column vector
        eigenvectors = eigenvectors.T 

        # sort eigenvectors w.r.t. eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # first k dimensions
        self.components = eigenvectors[:self.n_compoents]





    def transform(self, X):
        # project data
        X = X - self.mean

        return np.dot(X, self.components.T)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data =datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")

    plt.colorbar()
    plt.show()
    