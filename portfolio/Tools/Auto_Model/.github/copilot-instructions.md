# Copilot Instructions for Auto_Model Project

## Project Overview
The Auto_Model project automates machine learning model testing by providing individual evaluator classes for different models. Each evaluator tunes hyperparameters, evaluates performance, and returns the best model. A shared EvaluationMetrics class handles metric calculation and printing.

## Architecture
- **Major Components**: Model evaluators (e.g., LinearRegressorEval), EvaluationMetrics class, main script for orchestration
- **Service Boundaries**: Each model evaluator is self-contained; EvaluationMetrics provides shared evaluation functionality
- **Data Flows**: Data files → Model evaluator → Hyperparameter tuning → EvaluationMetrics → Best model output

## Developer Workflows
- **Build Process**: No build system; run Python scripts directly
- **Testing**: Manual testing by running evaluators; add unit tests as needed
- **Debugging**: Standard Python debugging; print statements in evaluators for hyperparameter tracking

## Conventions and Patterns
- **Code Style**: Follow PEP 8 for Python code
- **Naming**: Class names end with 'Eval' for evaluators; use descriptive variable names
- **Error Handling**: Handle file loading errors and invalid hyperparameters gracefully
- **Model Evaluator Pattern**: Each evaluator class has __init__ with data_files and hyperparams_range, and tune_and_evaluate method; includes standalone main block for testing
- **Standalone Execution**: Each evaluator file can be run directly with `python Models/ModelName_Eval.py`

## Dependencies and Integrations
- **External Dependencies**: scikit-learn, pandas, numpy
- **Integration Points**: Evaluators import EvaluationMetrics; main script imports all evaluators

## Key Files and Directories
- `Models/`: Directory containing all model evaluators and EvaluationMetrics
- `Models/Linear_Regressor_Eval.py`: Example linear regression evaluator
- `Models/Evaluation_Metrics.py`: Shared evaluation class
- `main.py`: Script to run multiple evaluators (to be created)
- `utils.py`: Shared utilities (to be created)