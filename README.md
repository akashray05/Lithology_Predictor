# Lithology Predictor

**Machine-learning-based lithology prediction from well-log data.**

Lithology Predictor is a machine-learning project for predicting subsurface lithology from geophysical well-log measurements. It combines a reproducible data-processing pipeline, engineered well-log features, multiclass classification models, model evaluation, and an interactive Streamlit application for exploring predictions and comparing model behavior.

The project uses the **FORCE 2020 well-log/lithology dataset** and evaluates models using well-level data splits to reduce information leakage between wells.

---

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Web Application](#web-application)
- [Machine-Learning Pipeline](#machine-learning-pipeline)
- [Dataset](#dataset)
- [Lithology Classes](#lithology-classes)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Evaluation Strategy](#evaluation-strategy)
- [Current Model Results](#current-model-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [Reproducibility and Data Integrity](#reproducibility-and-data-integrity)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Technology Stack](#technology-stack)
- [Dataset Citation and Acknowledgements](#dataset-citation-and-acknowledgements)
- [License](#license)

---

## Overview

Lithology identification is an important task in subsurface characterization and geological interpretation. Well-log measurements provide indirect information about formation properties, and interpreting these measurements can be challenging because different lithologies may exhibit overlapping log responses.

This project investigates supervised machine-learning methods for predicting lithology classes from well-log data.

The workflow covers:

1. Dataset inspection and exploratory analysis.
2. Data-quality assessment and preprocessing.
3. Well-level training, public evaluation, and blind-test separation.
4. Feature engineering and dataset construction.
5. Training and comparison of multiclass classification models.
6. Evaluation using accuracy, class-balanced metrics, and the FORCE penalty score.
7. Model packaging and integration into an interactive Streamlit application.
8. Visualization of well-log data, lithology predictions, and model performance.

The goal is to build a reproducible and transparent workflow that can support geological interpretation—not to replace geological expertise or independent validation.

## Project Objectives

- Develop a reliable machine-learning pipeline for well-log-based lithology prediction.
- Compare tree-based ensemble models under a common input-feature schema.
- Investigate the effect of class imbalance on lithology classification.
- Evaluate generalization using well-level splits rather than randomly mixing depth samples from the same wells.
- Preserve a strict boundary between model development and blind-test evaluation.
- Provide an interactive interface for examining input logs, predicted lithology, and model performance.
- Make the project easier to reproduce, inspect, and extend.

## Key Features

- **Multiclass classification:** Predicts among 12 FORCE 2020 lithology classes.
- **Multiple trained models:** ExtraTrees, XGBoost, and LightGBM.
- **Shared feature schema:** Uses the same 41-feature input schema across the three final models.
- **Well-level evaluation:** Separates training wells from public evaluation and blind-test wells.
- **Model comparison:** Supports comparison of classification performance and error patterns.
- **Class-aware evaluation:** Reports metrics beyond overall accuracy.
- **Interactive visualization:** Provides a Streamlit interface for exploring well-log data and model outputs.
- **Modular codebase:** Separates the application, model information, visualization utilities, tests, and notebooks.

---

## Web Application

The project includes an interactive web application built with **Streamlit**. Its canonical entry point is `final_app.py`.

The application is intended to make lithology prediction and model evaluation accessible through an interactive interface rather than requiring users to execute every analysis notebook manually.

### Application capabilities

The application and supporting modules are organized around the following functions.

| Area                    | Purpose                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Well-log exploration    | Inspect well-log measurements and their variation with depth.                             |
| Lithology visualization | Display lithology information alongside depth-oriented plots.                             |
| Model comparison        | Compare the behavior and evaluation results of the available models.                      |
| Prediction analysis     | Examine predicted lithology classes and their distribution.                               |
| Confusion analysis      | Investigate classification errors and agreement between predictions and reference labels. |
| Model information       | Present model details and available evaluation results.                                   |
| Bed-thickness analysis  | Examine contiguous lithology intervals and their thickness statistics.                    |
| Interactive plotting    | Explore plots and model results through a browser-based interface.                        |

The exact availability of each function depends on the data supplied and the corresponding application view.

### Planned application extensions

The following capabilities are intended extensions where not yet implemented or fully validated:

- Upload LAS and CSV files directly through the interface.
- Validate uploaded curves, depth columns, and missing values.
- Automatically apply the complete training-time preprocessing and feature-generation pipeline.
- Display per-sample prediction probabilities and uncertainty.
- Export predictions with depth and lithology codes.
- Download prediction tables as CSV.
- Provide additional well-log and lithology visualization controls.
- Offer a streamlined workflow for comparing predictions from multiple models.

### Application entry point

```bash
streamlit run final_app.py
```

---

## Machine-Learning Pipeline

The project follows a staged workflow so that data processing, feature generation, training, and evaluation remain traceable.

### 1. Dataset inspection

Inspect the input files, available curves, depth ranges, missing values, lithology labels, and class distributions.

### 2. Data-quality assessment

Identify missing or invalid measurements, inconsistent curve availability, potential outliers, and data-quality issues that may affect downstream predictions.

### 3. Preprocessing

Prepare the well-log measurements for analysis while maintaining a consistent representation of the input data.

### 4. Well-level splitting

Separate wells into three groups:

| Split             | Number of wells | Purpose                                           |
| ----------------- | --------------: | ------------------------------------------------- |
| Training          |              98 | Model development and fitting                     |
| Public evaluation |              10 | Model comparison and development-stage evaluation |
| Blind test        |              10 | Final evaluation on held-out wells                |

The official split manifest is stored at:

`data/splits/force2020_split.csv`

These partitions are defined at the well level to reduce leakage caused by neighboring depth samples from the same well appearing in different partitions.

### 5. Feature engineering

Generate a shared set of 41 input features from the available well-log measurements and related transformations.

The feature-generation pipeline is intended to ensure that all final models receive the same input schema.

### 6. Dataset construction

Assemble the model-ready features, depth information, well identifiers, and lithology targets while preserving the separation between training and evaluation wells.

### 7. Model training

Train and compare the selected tree-based ensemble classifiers.

### 8. Evaluation and error analysis

Assess overall performance, class-balanced performance, confusion patterns, and the project-specific FORCE penalty score.

### 9. Model packaging

Save trained estimators and supporting metadata for use by the application and evaluation workflows.

### 10. Interactive analysis

Present relevant data, predictions, model comparisons, and visualization tools through the Streamlit interface.

---

## Dataset

### FORCE 2020

The primary dataset is the FORCE 2020 well-log/lithology dataset, which provides well-log measurements and lithology labels for supervised learning.

The development workflow uses a collection of 118 LAS well files.

| Dataset property                          |     Value |
| ----------------------------------------- | --------: |
| LAS well files                            |       118 |
| Unique lithology classes                  |        12 |
| Labelled samples in the initial inventory | 1,431,383 |
| Shared final model input features         |        41 |
| Training wells                            |        98 |
| Public evaluation wells                   |        10 |
| Blind-test wells                          |        10 |

The sample count refers to the initial labelled-data inventory; counts may differ after filtering, preprocessing, or split-specific selection.

### Important data considerations

- Different wells may contain different subsets of logging curves.
- Lithology classes are imbalanced.
- Adjacent depth samples are correlated.
- Missing curves and missing measurements require consistent handling.
- A random sample-level split can overestimate generalization when measurements from the same well occur in both training and testing.
- Predictions depend on the available input measurements and the training-time feature-generation procedure.

The application should not be interpreted as a universal lithology classifier for arbitrary logs unless the input schema and preprocessing requirements are satisfied.

---

## Lithology Classes

The project predicts the following 12 FORCE 2020 lithology codes.

|    Code | Lithology       |
| ------: | --------------- |
| `30000` | Sandstone       |
| `65000` | Shale           |
| `65030` | Sandstone/Shale |
| `70000` | Limestone       |
| `70032` | Chalk           |
| `74000` | Dolomite        |
| `80000` | Marl            |
| `86000` | Anhydrite       |
| `88000` | Halite          |
| `90000` | Coal            |
| `93000` | Basement        |
| `99000` | Tuff            |

The mapping between lithology codes and class names must remain consistent across training, evaluation, prediction, visualization, and export.

---

## Feature Engineering

The final models use a shared **41-feature schema**.

Feature engineering is designed to extract useful information from well-log measurements while maintaining compatibility across the final estimators.

The broader workflow investigates:

- Original well-log measurements.
- Local rolling statistics and smoothing-related features.
- Depth-related information and local variation.
- Gradient or change-based features.
- Additional feature transformations evaluated during model development.

Not every experimental feature is necessarily part of the frozen 41-feature schema. The final feature list and order must be obtained from the model artifacts and the canonical feature-generation code.

### Feature consistency

For valid inference, the application must preserve:

1. The expected feature names.
2. The expected feature order.
3. The same transformations used during training.
4. Compatible handling of missing values.
5. Consistent lithology-code mapping.

Changing the feature-generation procedure without retraining and validating the affected models can produce unreliable predictions.

---

## Models

The project currently includes three final trained models.

| Model    | Implementation        | Role                        |
| -------- | --------------------- | --------------------------- |
| W5       | ExtraTrees classifier | Final tree-ensemble model   |
| XGBoost  | XGBoost classifier    | Gradient-boosted tree model |
| LightGBM | LightGBM classifier   | Gradient-boosted tree model |

The current model artifacts are stored under `models/`.

```text
models/
├── lithology_w5_final.joblib
├── lithology_xgboost_final.joblib
└── lithology_lightgbm_final.joblib
```

The shared feature schema allows the models to be compared under consistent input conditions.

Random Forest and CatBoost remain possible extensions unless separately trained and validated final artifacts are added to the project.

---

## Evaluation Strategy

Model evaluation considers more than overall accuracy because a model can achieve a high accuracy score while performing poorly on less frequent lithology classes.

### Metrics

- **Accuracy:** Fraction of predictions that match the reference labels.
- **Balanced accuracy:** Average recall across classes, reducing the dominance of frequent classes.
- **Macro F1-score:** Unweighted mean of the F1-score across classes.
- **Weighted F1-score:** F1-score averaged according to class support.
- **FORCE penalty score:** A task-specific metric used to assess classification performance according to the FORCE evaluation convention.
- **Confusion matrix:** Shows which lithology classes are confused with one another.

The FORCE penalty score should be interpreted using its exact scoring convention; in the reported results below, a less-negative value is better.

### Evaluation principles

- Keep the well-level split manifest fixed.
- Use training wells for model fitting.
- Use public evaluation wells for development-stage comparison.
- Reserve blind-test wells for final evaluation.
- Do not tune model settings based on blind-test results.
- Report class-balanced metrics alongside accuracy.
- Inspect per-class performance and confusion patterns.
- Preserve the same feature-generation pipeline across models.
- Clearly distinguish public evaluation results from blind-test results.

---

## Current Model Results

The following results are the reported public-evaluation metrics from the current project experiments.

| Metric              | W5 (ExtraTrees) | XGBoost | LightGBM |
| ------------------- | --------------: | ------: | -------: |
| Accuracy            |          73.87% |  73.82% |   70.53% |
| Balanced accuracy   |          46.37% |  46.60% |   48.22% |
| Macro F1-score      |          43.39% |  43.67% |   45.13% |
| Weighted F1-score   |          71.37% |  71.35% |   70.34% |
| FORCE penalty score |         -0.8745 | -0.9620 |  -0.9701 |

### Interpretation

- **W5 (ExtraTrees)** has the highest reported accuracy and the least-negative FORCE penalty score among these three models.
- **XGBoost** has accuracy close to W5, with slightly higher balanced accuracy and macro F1-score.
- **LightGBM** has the highest reported balanced accuracy and macro F1-score, indicating comparatively stronger average performance across classes.

These results illustrate a trade-off between overall accuracy and class-balanced performance. The preferred model depends on the intended evaluation objective.

**Important:** These figures are reported public-evaluation results, not a claim of blind-test performance. Blind-test results should be reported separately after the held-out evaluation has been completed and documented.

---

## Project Structure

The following is a representative overview of the current project organization. Some experimental notebooks and supporting files are omitted for readability.

```text
Lithology_Predictor/
├── final_app.py
├── UI_GUIDE.md
├── models/
│   ├── lithology_w5_final.joblib
│   ├── lithology_xgboost_final.joblib
│   └── lithology_lightgbm_final.joblib
├── data/
│   └── splits/
│       ├── force2020_split.csv
│       └── wells/
│           ├── train/
│           ├── public/
│           ├── blind_test/
│           └── split_manifest.csv
├── notebooks/
│   ├── 01_dataset_visualization.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_dataset_building.ipynb
│   ├── 06_model_baseline.ipynb
│   ├── 07_final_model.ipynb
│   ├── 08_final_evaluation.ipynb
│   ├── 09_gradient_features.ipynb
│   ├── 10_new_rolling_modes_and_class_specific.ipynb
│   ├── 11_class_specific_model.ipynb
│   ├── 12_final_models_exp.ipynb
│   ├── 13_final_test_of_models.ipynb
│   └── 14_blind_test_presentation_final.ipynb
├── src/
│   ├── __init__.py
│   ├── model_info.py
│   ├── plots_extra.py
│   ├── theme.py
│   └── well_stats.py
└── tests/
    ├── test_confusion_grid.py
    └── verify_ui.py
```

Additional project files, dependencies, configuration files, and data utilities may exist beyond this overview.

---

## Installation

### Requirements

- Python compatible with the project's installed dependencies.
- Git.
- A terminal or command prompt.
- Access to the required model artifacts and any data files needed by the selected application views.

### 1. Clone the repository

```bash
git clone https://github.com/akashray05/Lithology_Predictor.git
cd Lithology_Predictor
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency manifest is not available in the checked-out version, install the project's required dependencies according to the imports and documented environment. The application uses Streamlit, pandas, NumPy, Plotly, and the relevant model libraries.

The final model artifacts and required data files must also be available locally. Cloning the source repository alone does not guarantee that every large dataset or model artifact is included.

---

## Running the Application

From the repository root, activate the environment and run:

```bash
streamlit run final_app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser to interact with the application.

If the application fails to start, check the Python environment, installed dependencies, model artifact paths, and required input files.

For guidance on the interface, see [`UI_GUIDE.md`](UI_GUIDE.md).

---

## Using the Application

A typical workflow is:

1. Start the Streamlit application.
2. Open the relevant well-log or model-analysis view.
3. Select or provide the data supported by that view.
4. Inspect the available measurements and lithology information.
5. Review model predictions and the available evaluation plots.
6. Compare models using the supported metrics.
7. Investigate errors or class-specific behavior where reference labels are available.
8. Export results only through functionality exposed by the current application version.

The exact workflow depends on the data source and application view. Direct LAS/CSV upload, automated inference for arbitrary wells, and downloadable prediction files should be considered available only when implemented and verified in the deployed version.

---

## Reproducibility and Data Integrity

Reproducibility is a central principle of this project.

### Data leakage prevention

Well-log samples from the same well are strongly related. Randomly splitting individual samples can allow the model to learn well-specific patterns that also appear in the evaluation data.

The project therefore uses a well-level split manifest to separate training, public evaluation, and blind-test wells.

### Locked blind test

Blind-test wells must remain separate from model fitting, feature selection, hyperparameter tuning, and development decisions.

Their results should be used to estimate final performance on held-out wells, not repeatedly optimize the model.

### Shared preprocessing

All final models use a common 41-feature input schema. The training-time transformations and feature ordering must be preserved during inference.

### Traceability

For reproducible experiments, retain:

- The dataset source and version.
- The well split manifest.
- The feature-generation implementation.
- The trained model artifacts.
- The evaluation metrics and configuration.
- The Python environment and dependency versions.
- The mapping between lithology codes and class names.

### Testing

Run the project's tests with:

```bash
python -m pytest -q
```

The tests help check code behavior and supported interface components. Passing tests does not by itself establish geological validity or guarantee generalization to new wells.

---

## Limitations

- Lithology labels are imbalanced, and performance can differ substantially across classes.
- Similar log responses may correspond to different lithologies.
- Performance depends on the available logging curves, data quality, and preprocessing consistency.
- A model evaluated on FORCE 2020 wells may not generalize to other basins, logging tools, or acquisition conditions without further validation.
- The reported public-evaluation metrics do not substitute for independent blind-test results.
- Predicted lithology is a model estimate and should be interpreted alongside geological context and independent evidence.
- A functioning local Streamlit application does not automatically imply that a public deployment is available.
- Features described as planned are not guaranteed to exist in the current application.

---

## Roadmap

### Completed or established

- [x] FORCE 2020 dataset inspection and exploratory analysis.
- [x] Data-quality and preprocessing investigations.
- [x] Well-level train/public/blind-test split manifest.
- [x] Feature-engineering experiments.
- [x] Shared 41-feature schema for the final models.
- [x] Final ExtraTrees, XGBoost, and LightGBM model artifacts.
- [x] Public-evaluation metrics and model comparison.
- [x] Streamlit application entry point at `final_app.py`.
- [x] Supporting visualization and model-information modules.
- [x] Automated tests for selected application components.

### Future improvements

- [ ] Publish a verified, reproducible dependency manifest.
- [ ] Document the complete feature schema and expected input curves.
- [ ] Expand per-class error analysis and calibration assessment.
- [ ] Complete and document blind-test reporting.
- [ ] Add or refine LAS/CSV upload and input validation.
- [ ] Provide a consistent prediction export format.
- [ ] Add prediction-probability and uncertainty visualizations where appropriate.
- [ ] Evaluate additional algorithms, including Random Forest and CatBoost.
- [ ] Investigate sequence-aware methods for depth-continuous predictions.
- [ ] Add model and dataset version metadata to exported results.
- [ ] Deploy a public version of the Streamlit application, if desired.
- [ ] Improve documentation and reproducibility for external users.

---

## Technology Stack

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Main programming language                 |
| pandas           | Tabular data processing                   |
| NumPy            | Numerical computation                     |
| scikit-learn     | Machine-learning utilities and ExtraTrees |
| XGBoost          | Gradient-boosted tree classification      |
| LightGBM         | Gradient-boosted tree classification      |
| Plotly           | Interactive visualization                 |
| Streamlit        | Web application                           |
| Jupyter Notebook | Exploratory analysis and experiments      |
| pytest           | Automated testing                         |
| Git and GitHub   | Version control and project hosting       |

---

## Dataset Citation and Acknowledgements

This project uses the FORCE 2020 well-log/lithology dataset for model development and evaluation.

Please consult the original FORCE 2020 dataset publication and its official distribution for the authoritative dataset citation, usage conditions, and licensing requirements. Dataset attribution and licensing should be preserved when redistributing derived materials.

The project is intended for research, educational, and technical exploration of machine-learning methods in geoscience.

---

## License

A license has not been specified in this README. Before redistributing or reusing the project, add an appropriate repository license and verify that it is compatible with the dataset and any third-party model dependencies.

---

**Lithology Predictor — exploring machine learning for well-log interpretation.**
