from sklearnex import patch_sklearn 
patch_sklearn()

from deepview import DeepView
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
# from sklearn.datasets import fetch_openml
# make bolb
from sklearn.datasets import make_blobs

clf = svm.SVC(probability=True)

## get minist data

# mnist = fetch_openml('mnist_784', version=1, cache=True)
# mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
# X, y = mnist["data"], mnist["target"]

X, y = make_blobs(n_samples=10000, centers=10, n_features=784, random_state=0)

X_train = X[:5000] 
y_train = y[:5000]

clf = clf.fit(X_train, y_train)

dv = DeepView(pred_fn=clf.predict_proba, classes=set(y), max_samples=5000, batch_size=1000, data_shape=X.shape[1:],)
dv.add_samples(X_train, y_train)