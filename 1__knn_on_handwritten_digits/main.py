from typing import *
import numpy as np
from sklearn.datasets import fetch_openml
from knn import KNeighborsClassifier #from sklearn.neighbors import KNeighborsClassifier
import os


DATASET_CACHE_DIR = './scikit_learn_data'
RESULTS_DIR = './results'
PART_C_CSV = f'{RESULTS_DIR}/part_c.csv'
PART_D_CSV = f'{RESULTS_DIR}/part_d.csv'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Generating predictor...")
    predictor = Predictor()
    print("Done")

    print("Part (c):")
    results_c = open(PART_C_CSV, 'w')
    results_c.write('n,k,accuracy\n')
    for k in range(1, 100+1):
        n, k = 1000, k
        prediction_accuracy = predictor.calculate_accuracy(n=n, k=k)
        print(f"Prediction accuracy for n=1000, k={k} is: {prediction_accuracy}")
        results_c.write(f'{n},{k},{prediction_accuracy}\n')
    results_c.close()

    print("Part (d):")
    results_d = open(PART_D_CSV, 'w')
    results_d.write('n,k,accuracy\n')
    for i in range(1,50+1):
        n, k = i*100, 1
        prediction_accuracy = predictor.calculate_accuracy(n=n, k=k)
        print(f"Prediction accuracy for n={n}, k=1 is: {prediction_accuracy}")
        results_d.write(f'{n},{k},{prediction_accuracy}\n')
    results_d.close()


class Predictor:
    def __init__(self, data_set_limit: int = None):
        self.data_set_limit = data_set_limit
        self.data, self.labels = None, None
        self.idx, self.train_idx, self.test_idx = None, None, None
        self.train, self.train_labels, self.test, self.test_labels = None, None, None, None
        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
        self._load_data()
        self._generate_idx()
        self._load_train_test()
    
    def _load_data(self):
        print("_load_data() - START")
        mnist = fetch_openml('mnist_784', as_frame=False, data_home=DATASET_CACHE_DIR)
        self.data = mnist['data'][:self.data_set_limit]
        self.labels = mnist['target'][:self.data_set_limit]
        print("_load_data() - END")
    
    def _generate_idx(self):
        if self.data_set_limit is None:
            range_end, idx_size = 70000, 11000
            cutoff = 10000
        else:
            range_end, idx_size = self.data_set_limit, int(self.data_set_limit*(0.5))
            cutoff = int(idx_size*(10/11))
        self.idx = np.random.RandomState(0).choice(range_end, idx_size)
        self.train_idx, self.test_idx = self.idx[:cutoff], self.idx[cutoff:]
    
    def _load_train_test(self):
        self.train = self.data[self.train_idx, :].astype(int)
        self.train_labels = self.labels[self.train_idx]
        self.test = self.data[self.test_idx, :].astype(int)
        self.test_labels = self.labels[self.test_idx]

    def predict_query_label(self, n: int, k: int, query_image: np.ndarray) -> str:
        """
        Uses k-NN algorithm to predict the label of query_image, using the first n datapoints from the provided training set
        """
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(self.train[:n], self.train_labels[:n])
        label = classifier.predict(query_image)
        return label
    
    def calculate_accuracy(self, n: int, k: int) -> float:
        """
        This returns prediction accuracy of classifier using first n training images and k neighborhoods, as %
        Uses k-NN algorithm to predict the label of each query image
        """
        correct_predictions: int = 0
        predict_labels: List[str] = self.predict_query_label(n, k, self.test)
        correct_predictions: int = np.sum(predict_labels == self.test_labels)
        prediction_accuracy: float = correct_predictions / len(self.test)
        return prediction_accuracy
    

if __name__ == "__main__":
    main()