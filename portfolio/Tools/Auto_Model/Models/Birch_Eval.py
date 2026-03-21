import os
import pandas as pd
from sklearn.cluster import Birch
from Clustering_Metrics import ClusteringMetrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class BirchEval:
    def __init__(self, data_files, hyperparams_range):
        self.data_files = data_files
        self.hyperparams_range = hyperparams_range
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None
        self.evaluator = ClusteringMetrics()

    def load_data(self):
        self.X_train = pd.read_csv(self.data_files['train']).drop('target', axis=1, errors='ignore')
        self.X_test = pd.read_csv(self.data_files['test']).drop('target', axis=1, errors='ignore')
        self.X_valid = pd.read_csv(self.data_files['valid']).drop('target', axis=1, errors='ignore')

    def tune_and_evaluate(self):
        self.load_data()

        from itertools import product
        param_names = list(self.hyperparams_range.keys())
        param_values = list(self.hyperparams_range.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            model = Birch(**params)
            labels = model.fit_predict(self.X_train)
            score = self.evaluator.evaluate_and_print(model, self.X_train, labels, params)

            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_params = params

        test_labels = self.best_model.predict(self.X_test)
        final_score = self.evaluator.evaluate_and_print(self.best_model, self.X_test, test_labels, self.best_params)
        return self.best_model, self.best_params, final_score


if __name__ == "__main__":
    data_files = {
        'train': os.path.join(PROJECT_ROOT, 'train_classification.csv'),
        'test': os.path.join(PROJECT_ROOT, 'test_classification.csv'),
        'valid': os.path.join(PROJECT_ROOT, 'valid_classification.csv')
    }

    hyperparams_range = {
        'n_clusters': [2, 3, 4, 5],
        'threshold': [0.3, 0.5, 0.7],
        'branching_factor': [25, 50, 100]
    }

    evaluator = BirchEval(data_files, hyperparams_range)
    best_model, best_params, score = evaluator.tune_and_evaluate()

    print(f"\nFinal Results:")
    print(f"Best score: {score}")
    print(f"Best parameters: {best_params}")
