# AML Challenge 2024

## Authors
- Etienne Roulet
- Alexander Shanmugam

## Table of Contents
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Saved Models and Cross-Validation Predictions](#saved-models-and-cross-validation-predictions)

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Notebook

1. **Activate the virtual environment:**
    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Run Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

3. **Open the notebook:**
    Open `AML_MC.ipynb` from the Jupyter Notebook interface.

## Project Structure

- `data/`: Contains the data files.
- `notebooks/`: Contains Jupyter notebooks.
- `saved_models/`: Directory where trained models and cross-validation predictions are saved.
- `requirements.txt`: Lists the Python packages required to run the project.
- `README.md`: This file.

## How It Works

The project is designed to develop and evaluate affinity models for personalized credit card marketing campaigns for a bank. The primary tasks include:

1. **Data Preprocessing:**
    - Load and clean the data.
    - Perform exploratory data analysis (EDA).
    - Feature engineering and selection.
    - Handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).

2. **Modeling:**
    - Define and train multiple models.
    - Use cross-validation for model evaluation.
    - Save the trained models and cross-validation predictions.

3. **Model Evaluation:**
    - Plot ROC curves and confusion matrices.
    - Compare top N customers identified by each model.
    - Display benchmark results.

4. **Model Explainability:**
    - Use LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) for model interpretation.

## Saved Models and Cross-Validation Predictions

The models and cross-validation predictions are saved in the `saved_models/` directory. Here's how it works:

- **Saving Models:**
    - After training a model, it is saved using `joblib` in the `saved_models/` directory.
    - The filename format is `<model_name>.pkl`.

- **Saving Cross-Validation Predictions:**
    - Cross-validation predictions are saved in the `saved_models/` directory with the filename format `<model_name>_cv_preds.pkl`.

- **Loading Models:**
    - The script checks for existing models in the `saved_models/` directory.
    - If a model is found, it is loaded to avoid retraining.

- **Retraining:**
    - The script ensures that only models not previously trained are retrained.

By following these instructions, you should be able to set up the environment, run the notebook, and understand the workflow of the project.
