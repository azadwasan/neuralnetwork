from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# X and y must be converted to list, otherwise we wouldn't be able to extract them
# Earlier X and y were being returned without conversion and the extraction in C++ failed,
# that is because the C++ code assumes these to be lists, where as they actually are <class 'numpy.ndarray'>
def getRegressionData(nSamples, nFeatures, noise):
    X, y = make_regression(n_samples=nSamples, n_features=nFeatures, noise=noise)
    return X.tolist(), y.tolist()   


def getClassificationData(numSamples, numFeatures, redundantFeatures, clustersPerClass, randomState = 42):
    informativeFeatures = numFeatures - redundantFeatures        
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=numSamples,
        n_features=numFeatures,
        n_informative=informativeFeatures,
        n_redundant=redundantFeatures,
        n_clusters_per_class=clustersPerClass,
        random_state=randomState
    )
    # Convert to list as X and y are Array type
    return X.tolist(), y.tolist()

