# MLflow Experiment Tracking and Model Versioning

This project demonstrates how to use MLflow for:

- Experiment tracking (parameters, metrics)
- Artifact logging (trained datasets, models)
- Model versioning (with MLflow Model Registry)

We use the **Iris dataset** and train **Random Forest Classifiers** with different hyperparameters.

## Files

- `train.py` — Code to train a model (logs parameters, metrics, and model to MLflow)
- `requirements.txt` — Python libraries required
- `README.md` — This file

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Start MLflow UI:
mlflow ui

3. Run the training script:
python train.py
