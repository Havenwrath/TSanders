# Auto_Model

A project for automated machine learning model testing and evaluation.

## Overview

Auto_Model provides individual evaluator classes for different machine learning models. Each evaluator automatically tunes hyperparameters, evaluates performance using a shared evaluation metrics class, and returns the best performing model.

## Features

- Automated hyperparameter tuning using grid search
- Shared evaluation metrics for consistent reporting
- Support for multiple model types
- Easy comparison of model performance

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running a Single Model Evaluator

Each model evaluator can be run standalone:

```bash
cd Models
python Linear_Regressor_Eval.py
python Random_Forest_Regressor_Eval.py
```

Or import and use programmatically:

```python
from Models.Linear_Regressor_Eval import LinearRegressorEval

data_files = {
    'train': 'train.csv',
    'test': 'test.csv',
    'valid': 'valid.csv'
}

hyperparams_range = {
    'fit_intercept': [True, False]
}

evaluator = LinearRegressorEval(data_files, hyperparams_range)
best_model, best_params, score = evaluator.tune_and_evaluate()

print(f"Best score: {score}")
print(f"Best parameters: {best_params}")
```

### Running Multiple Models

Use `main.py` to run and compare multiple models:

```bash
python main.py
```

## Project Structure

```
Auto_Model/
├── Models/
│   ├── Linear_Regressor_Eval.py
│   ├── Random_Forest_Regressor_Eval.py
│   ├── Evaluation_Metrics.py
│   └── ... (other model evaluators)
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

## Adding New Model Evaluators

1. Create a new file in the `Models/` directory (e.g., `NewModel_Eval.py`)
2. Implement a class following the pattern of existing evaluators
3. Import and use `EvaluationMetrics` for consistent evaluation
4. Update `main.py` to include the new evaluator

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Test your changes thoroughly
4. Update documentation as needed

## License

[Add your license here]