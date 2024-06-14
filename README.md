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



## Chatgpt Promting Strategie: 250-500 Wörter:

### Verwendung von ChatGPT in diesem Projekt

Die Verwendung von ChatGPT oder vergleichbaren AI-Tools war in diesem Projekt erlaubt und wurde als Unterstützung bei der Generierung von Code-Snippets, der Erstellung von Kennzahlen und der Analyse von Metriken verwendet. Die nachfolgenden Abschnitte erläutern, für welche Aufgaben das AI-Tool eingesetzt wurde, welche Prompting-Strategien angewendet wurden und welche Strategie am erfolgreichsten war.

#### Einsatzgebiete des AI-Tools

1. **Generierung von Code-Snippets**:
   - ChatGPT wurde zur Generierung von Pseudo-Code verwendet, um bestimmte Aufgaben zu automatisieren. Dies beinhaltete beispielsweise die Datenvorverarbeitung, die Erstellung von Visualisierungen und die Implementierung von Modellierungsansätzen. Durch die Bereitstellung von Code-Vorlagen konnte der Entwicklungsprozess beschleunigt werden.
   
2. **Analyse von Kennzahlen und Metriken**:
   - Das Tool wurde auch zur Analyse von verschiedenen Kennzahlen und Metriken verwendet. Es half dabei, die Leistungsfähigkeit der Modelle zu bewerten und die Ergebnisse in einer strukturierten Form darzustellen. ChatGPT unterstützte bei der Berechnung und Visualisierung von Performance-Metriken wie Genauigkeit, Präzision, Recall und F1-Score.

3. **Erklärung und Dokumentation**:
   - Zusätzlich half ChatGPT bei der Erstellung von erklärenden Texten und Dokumentationen, um die durchgeführten Schritte und Ergebnisse nachvollziehbar darzustellen. Dies umfasste die Beschreibung der Daten, die durchgeführten Transformationen und die Ergebnisse der Modellierung.

#### Prompting-Strategien

1. **Direkte Code-Anfragen**:
   - Bei spezifischen Aufgabenstellungen wurden direkte Anfragen an ChatGPT gestellt, um Code für bestimmte Aufgaben zu generieren. Zum Beispiel: "Erstelle eine Funktion zur Datenvorverarbeitung" oder "Generiere eine Visualisierung der Verteilung der Darlehensbeträge".
   
2. **Iterative Verfeinerung**:
   - Eine iterative Strategie wurde angewendet, bei der initiale Code-Snippets von ChatGPT erstellt und dann manuell verfeinert wurden. Durch wiederholtes Feedback und Anpassungen konnte der generierte Code schrittweise verbessert und auf die spezifischen Anforderungen des Projekts zugeschnitten werden.

3. **Analyse und Interpretation**:
   - Für die Analyse und Interpretation von Ergebnissen wurden spezifische Fragen formuliert, um detaillierte Antworten zu erhalten. Zum Beispiel: "Wie hoch ist die Genauigkeit des Modells?" oder "Welche Metriken sollten zur Bewertung der Modellleistung verwendet werden?".

#### Erfolgreichste Prompting-Strategie

Die erfolgreichste Prompting-Strategie war die **iterative Verfeinerung**. Diese Strategie ermöglichte es, initialen Code von ChatGPT schnell zu erhalten und dann durch menschliche Expertise zu verfeinern. Diese Kombination aus AI-generierter Basis und menschlicher Anpassung führte zu effizienteren und genaueren Ergebnissen.

**Vorteile dieser Strategie**:
- **Effizienz**: Schnellere Generierung von initialem Code, der als Grundlage für weitere Anpassungen dient.
- **Präzision**: Möglichkeit, den generierten Code schrittweise zu verbessern und auf die spezifischen Anforderungen des Projekts zuzuschneiden.
- **Flexibilität**: Anpassungen und Verfeinerungen konnten flexibel und iterativ durchgeführt werden, was zu besseren Endergebnissen führte.

**Nachteile**:
- **Abhängigkeit von menschlicher Expertise**: Die Qualität des Endergebnisses hing stark von der Fähigkeit ab, den generierten Code effektiv zu überprüfen und anzupassen.
- **Zeitaufwand**: Obwohl der initiale Code schnell generiert wurde, erforderte die iterative Verfeinerung dennoch einen gewissen Zeitaufwand.

Insgesamt hat sich die iterative Verfeinerung als die effektivste Strategie erwiesen, um das AI-Tool in diesem Projekt als Unterstützung zu nutzen. Die Kombination aus automatisierter Generierung und menschlicher Anpassung ermöglichte es, die Aufgaben effizient und präzise zu erledigen.