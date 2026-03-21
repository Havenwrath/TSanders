from sklearn.metrics import mean_squared_error, r2_score

class EvaluationMetrics:
    def evaluate_and_print(self, model, X_test, y_test, hyperparams):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Hyperparameters: {hyperparams}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print("-" * 50)

        return r2  # Return score for comparison (higher is better for R2)