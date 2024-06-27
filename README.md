# Oropharyngeal-Cancer-Prediction-

# Oropharyngeal Cancer Detection with Machine Learning

## Table of Contents
- [Overview](#overview)
- [Algorithms Used](#algorithms-used)
  - [Logistic Regression-Decision Tree](#logistic-regression-decision-tree)
  - [K-Nearest Neighbors-Decision Tree](#k-nearest-neighbors-decision-tree)
  - [Naïve Bayes-Random Forest](#naïve-bayes-random-forest)
  - [Support Vector Machine-Decision Tree](#support-vector-machine-decision-tree)
  - [Support Vector Machine-Random Forest](#support-vector-machine-random-forest)
  - [Ensemble Model: Random Forest, Gradient Boosting, Extra Trees, and SVC](#ensemble-model-random-forest-gradient-boosting-extra-trees-and-svc)
- [Methodology](#methodology)
  - [Architecture](#architecture)
  - [Dataset](#dataset)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

## Overview
This project explores the application of machine learning (ML) techniques for detecting oropharyngeal cancer. By leveraging advanced ML models, the goal is to improve early detection accuracy and assist in personalized treatment planning.

## Algorithms Used
### Logistic Regression-Decision Tree
Combines logistic regression for linear relationships with decision trees for capturing non-linear patterns.

### K-Nearest Neighbors-Decision Tree
Integrates KNN for local insights with decision trees for global pattern recognition.

### Naïve Bayes-Random Forest
Pairs Naïve Bayes' simplicity with Random Forest's robustness for enhanced classification.

### Support Vector Machine-Decision Tree
Utilizes SVM for high-dimensional data and decision trees for complex relationship modeling.

### Support Vector Machine-Random Forest
Combines SVM's resilience with Random Forest's generalization capabilities.

### Ensemble Model: Random Forest, Gradient Boosting, Extra Trees, and SVC
An ensemble approach using multiple classifiers to achieve high accuracy and robustness.

## Methodology
### Architecture
Standard preprocessing (SMOTE oversampling, label encoding, handling missing values) and feature selection (ensemble classifiers, XGBoost).

### Dataset
Utilizes a dataset of 3,346 CT image volumes of head and neck cancer, ideal for ML model training and evaluation.

### Model Training and Evaluation
Models trained using cross-validation (70-30 split) to assess accuracy, precision, recall, and F1-score.

## Results
Achieved high accuracies across hybrid and ensemble models, demonstrating effectiveness in oropharyngeal cancer detection.

## Usage
Clone the repository and follow instructions in `README.md` to run models on your dataset.

## Contributions
Contributions welcome! Fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
