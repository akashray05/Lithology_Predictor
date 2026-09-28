# Lithology Predictor

Machine-learning system for predicting lithology from well-log data.

## Project Objective

The goal of this project is to develop a reproducible machine-learning
pipeline that takes well-log measurements and predicts lithology along depth.

The project will investigate multiple supervised multiclass classification
algorithms and eventually provide an interactive web application for
well-log-based lithology prediction.

## Planned Models

- Random Forest
- Extra Trees
- XGBoost
- CatBoost

## Dataset

The primary dataset for development is the FORCE 2020 well-log/lithology
dataset.

## Planned Workflow

1. Dataset inspection
2. Exploratory data analysis
3. Data cleaning
4. Well-level train/validation/test splitting
5. Feature engineering
6. Baseline models
7. Model comparison
8. Hyperparameter optimization
9. Final evaluation
10. Model packaging
11. Web application
12. Interactive lithology visualization

## Planned Web Application

The application will eventually support:

- LAS upload
- CSV upload
- Input validation
- Automatic preprocessing
- Lithology prediction
- Prediction probabilities
- Interactive well-log plots
- Lithology-versus-depth visualization
- Model comparison
- Prediction download

## Project Principles

- Reproducible preprocessing
- No data leakage
- Well-level evaluation
- Locked test set
- One shared preprocessing/feature-generation pipeline
- Explicit model and dataset metadata
- Evaluation using class-balanced metrics, not accuracy alone

## Status

Early project initialization.

The ML pipeline and web application have not yet been implemented.
