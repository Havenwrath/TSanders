import os
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV
from Evaluation_Metrics import EvaluationMetrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class HuberRegressorEval:
    def __init__(self, data_files, hyperparams_range):
        self.data_files = data_files
        self.hyperparams_range = hyperparams_range
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None
        self.evaluator = EvaluationMetrics()

    def load_data(self):
        self.X_train = pd.read_csv(self.data_files['train']).drop('target', axis=1)
        self.y_train = pd.read_csv(self.data_files['train'])['target']
        self.X_test = pd.read_csv(self.data_files['test']).drop('target', axis=1)
        self.y_test = pd.read_csv(self.data_files['test'])['target']
        self.X_valid = pd.read_csv(self.data_files['valid']).drop('target', axis=1)
        self.y_valid = pd.read_csv(self.data_files['valid'])['target']

    def tune_and_evaluate(self):
        self.load_data()

        model = HuberRegressor()
        grid_search = GridSearchCV(model, self.hyperparams_range, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)

        for params, mean_score, _ in zip(grid_search.cv_results_['params'],
                                        grid_search.cv_results_['mean_test_score'],
                                        grid_search.cv_results_['std_test_score']):
            model.set_params(**params)
            model.fit(self.X_train, self.y_train)
            score = self.evaluator.evaluate_and_print(model, self.X_valid, self.y_valid, params)

            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_params = params

        final_score = self.evaluator.evaluate_and_print(self.best_model, self.X_test, self.y_test, self.best_params)
        return self.best_model, self.best_params, final_score


if __name__ == "__main__":
    data_files = {
        'train': os.path.join(PROJECT_ROOT, 'train.csv'),
        'test': os.path.join(PROJECT_ROOT, 'test.csv'),
        'valid': os.path.join(PROJECT_ROOT, 'valid.csv')
    }

    hyperparams_range = {
        'epsilon': [1.1, 1.35, 1.5, 2.0],
        'alpha': [0.0001, 0.001, 0.01]
    }

    evaluator = HuberRegressorEval(data_files, hyperparams_range)
    best_model, best_params, score = evaluator.tune_and_evaluate()

    print(f"\nFinal Results:")
    print(f"Best score: {score}")
    print(f"Best parameters: {best_params}")
