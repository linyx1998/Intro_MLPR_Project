import numpy as np
from sklearn import svm
from read import read_data
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import MDS
import sys

def run_svm(X_train_, y_train, X_test_, y_test):
    for c in (0.001, 0.01, 0.1, 1, 10, 100):
        clf = svm.SVC(kernel='rbf', C=c).fit(X_train_, y_train)
        print('rbf',c,"=> train acc:",np.mean(clf.predict(X_train_) == y_train),\
            "test acc:",np.mean(clf.predict(X_test_) == y_test))

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = read_data(True)
    # X_train = X_train[:1000]
    # y_train = y_train[:1000]
    # X_test = X_test[:1000]
    # y_test = y_test[:1000]
    feature_remain_ratio = np.linspace(0.1,0.9,8) # from 50% to 90%
    feature_remain = (feature_remain_ratio * X_train[0].size).astype(int);

    if(sys.argv[1]=='PCA'):
        print('PCA')
        for r in range(feature_remain.size):
            print('feature remain %.2f' % feature_remain_ratio[r]+'*'+str(X_train[0].size)+\
                '='+str(feature_remain[r]))
            pca = PCA(feature_remain[r])
            X_train_ = pca.fit_transform(X_train)
            X_test_ = pca.fit_transform(X_test)
            run_svm(X_train_, y_train, X_test_, y_test)
            print()
    elif(sys.argv[1]=='KernelPCA'):
        print('KernelPCA')
        for r in range(feature_remain.size):
            print('feature remain %.2f' % feature_remain_ratio[r]+'*'+str(X_train[0].size)+\
                '='+str(feature_remain[r]))
            pca = KernelPCA(kernel="rbf", n_components=feature_remain[r])
            X_train_ = pca.fit_transform(X_train)
            X_test_ = pca.fit_transform(X_test)
            run_svm(X_train_, y_train, X_test_, y_test)
            print()
    elif(sys.argv[1]=='LLE'):
        print('LLE')
        for r in range(feature_remain.size):
            print('feature remain %.2f' % feature_remain_ratio[r]+'*'+str(X_train[0].size)+\
                '='+str(feature_remain[r]))
            X_train_ = LLE(n_neighbors=30, n_components=feature_remain[r], method='standard').\
                fit_transform(X_train)
            X_test_ = LLE(n_neighbors=30, n_components=feature_remain[r], method='standard').\
                fit_transform(X_test)
            run_svm(X_train_, y_train, X_test_, y_test)
            print()