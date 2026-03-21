# AGENTS.md

## Commands
- Install: `pip install -r requirements.txt`
- Run evaluator standalone: `python Models/<Model>_Eval.py` (e.g., `python Models/Linear_Regressor_Eval.py`)
- Run all models: `python main.py`
- No test framework configured; add pytest for unit tests

## Architecture
- `Models/` - Model evaluators (e.g., `Linear_Regressor_Eval.py`) and shared `Evaluation_Metrics.py`
- `auto_model/` - Package with `model_tester.py`, `models/`, `utils/`, `examples/`
- `main.py` - Orchestrates running multiple evaluators
- Data files: `train.csv`, `test.csv`, `valid.csv` with `target` column

## Code Style
- Follow PEP 8
- Class names: `<Model>Eval` (e.g., `LinearRegressorEval`)
- Each evaluator has `__init__(data_files, hyperparams_range)` and `tune_and_evaluate()` method
- Include standalone `if __name__ == "__main__":` block for direct execution
- Imports: pandas, sklearn, numpy; use relative imports for `Evaluation_Metrics`
- Handle file loading errors and invalid hyperparameters gracefully
