from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from subprocess import PIPE, CalledProcessError
import numpy as np
import os
import subprocess
import tempfile

class KnnClassifier(BaseEstimator, ClassifierMixin):
    """
    Knn multiclass classifier

    It also runs PCA to reduce dimensionality

    Parameters
    ----------
    knn_path : str, optional
        The path to the knn executable program
    k : int, optional
        The number of neighbours to look for in the Knn algorithm
    alpha : int, optional
        The number of principal components to use for Knn
    n_iter : int, optional
        The number of iterations of the power method to get all eigenvalues
        when doing PCA.
    with_pca : bool, optional
        Whether to use PCA or not
    quiet : bool, optional
        Be quiet (do not print stderr of knn executable)

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, knn_path='tp2', k=3, alpha=37, n_iter=1000, with_pca=True, quiet=False):
        self.knn_path = knn_path
        self.k = k
        self.alpha = alpha
        self.with_pca = with_pca
        self.n_iter = n_iter
        self.quiet = quiet

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        classes = list(self.classes_)
        self.X_ = X
        self.y_ = np.array([classes.index(y_i) for y_i in y])

        # Return the classifier
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        oldX = X
        X = check_array(X)

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('invalid number of features: ' \
                    'expected {} but was {}'.format(self.X_.shape[1], X.shape[1]))

        # Run knn program with X
        y = self._run_knn(X)

        return self.classes_[y]

    def _write_train_data_to_temp_file(self):
        array = np.column_stack((self.y_, self.X_))
        return self._write_csv_to_temp_file(array)

    def _write_test_data_to_temp_file(self, X):
        return self._write_csv_to_temp_file(X)

    def _write_csv_to_temp_file(self, array):
        with tempfile.NamedTemporaryFile(delete=False) as tp:
            np.savetxt(tp, array, fmt='%d',
                delimiter=',', header='foo')
            tp.close()
            return tp.name

    def _run_knn(self, X):
        # Prepare train and test data
        train_path = self._write_train_data_to_temp_file()
        test_path = self._write_test_data_to_temp_file(X)

        # Create a temporary file for output predictions
        classif_f = tempfile.NamedTemporaryFile(delete=False)
        classif_f.close()
        classif_path = classif_f.name

        def unlink_temp_files():
            os.unlink(train_path)
            os.unlink(test_path)
            os.unlink(classif_path)

        # Prepare command arguments
        args = [
            self.knn_path,
            '-m', '1' if self.with_pca else '0',
            '-i', train_path,
            '-q', test_path,
            '-o', classif_path
        ]

        # Define parameters
        env = os.environ.copy()
        env["K"] = str(self.k)
        env["ALPHA"] = str(self.alpha)
        env["N_ITER"] = str(self.n_iter)

        # Run process and process results
        proc = subprocess.run(' '.join(args), shell=True, stderr=PIPE, env=env)

        if proc.returncode > 0:
            print(proc.stderr.decode())
            unlink_temp_files()
            raise RuntimeError('{}: non-zero exit status'.format(args))

        if not self.quiet:
            print(proc.stderr.decode())

        results = np.loadtxt(classif_path, dtype=np.uint8, delimiter=',', skiprows=1)
        unlink_temp_files()
        return results[:, 1]

    def debug(self, X):
        train_path = self._write_train_data_to_temp_file()
        test_path = self._write_test_data_to_temp_file(X)
        return train_path, test_path


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator
    import os

    # Include path to binaries on PATH variable
    bin_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    os.environ['PATH'] = "{}:{}".format(bin_path, os.environ['PATH'])

    #check_estimator(KnnClassifier)
    #print("Classifier is OK!")

    from exp import *
    X, y = load_data('data/train.csv')

    X = X[:8000,:]
    y = y[:8000]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    import time
    ts = time.time()
    clf = KnnClassifier(k=3, alpha=37, quiet=False)
    clf.fit(X_train, y_train)
    #print(clf.debug(X_test))

    res = clf.score(X_test, y_test)
    te = time.time()
    print(res)
    print("Took {} seconds".format(te - ts))
