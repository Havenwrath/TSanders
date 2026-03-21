# Main script to run model evaluations
from Models.Linear_Regressor_Eval import LinearRegressorEval
from Models.Random_Forest_Regressor_Eval import RandomForestRegressorEval

def main():
    # Example usage
    data_files = {
        'train': 'train.csv',
        'test': 'test.csv',
        'valid': 'valid.csv'
    }

    # Linear Regression hyperparameters
    lr_hyperparams = {
        'fit_intercept': [True, False]
    }

    # Random Forest hyperparameters
    rf_hyperparams = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20]
    }

    print("Evaluating Linear Regression...")
    lr_evaluator = LinearRegressorEval(data_files, lr_hyperparams)
    lr_model, lr_params, lr_score = lr_evaluator.tune_and_evaluate()

    print("\nEvaluating Random Forest...")
    rf_evaluator = RandomForestRegressorEval(data_files, rf_hyperparams)
    rf_model, rf_params, rf_score = rf_evaluator.tune_and_evaluate()

    print("
Comparison:")
    print(f"Linear Regression - Score: {lr_score:.4f}, Params: {lr_params}")
    print(f"Random Forest - Score: {rf_score:.4f}, Params: {rf_params}")

if __name__ == "__main__":
    main()