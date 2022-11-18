import numpy as np
from sklearn import svm
from read import read_data

if __name__ == '__main__':
    X_train, y_train, x_test, y_test = read_data(True)

    for k in ('linear', ['rbf']):
        for c in (0.0001, 0.001, 0.01, 0.1, 1):
            clf = svm.SVC(kernel=k, C=c).fit(X_train, y_train)
            print(k,c,"=> train acc:",np.mean(clf.predict(X_train) == y_train),\
                "test acc:",np.mean(clf.predict(x_test) == y_test))