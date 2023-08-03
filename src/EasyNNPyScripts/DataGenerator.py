from sklearn.datasets import make_regression

# X and y must be converted to list, otherwise we wouldn't be able to extract them
# Earlier X and y were being returned without conversion and the extraction in C++ failed,
# that is because the C++ code assumes these to be lists, where as they actually are <class 'numpy.ndarray'>
def getRegressionData(nSamples, nFeatures, noise):
    X, y = make_regression(n_samples=nSamples, n_features=nFeatures, noise=noise)
    return X.tolist(), y.tolist()   
