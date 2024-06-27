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
Oropharyngeal cancer poses significant challenges in healthcare due to its impact on critical areas like the throat, tonsils, and base of the tongue. Traditional diagnostic methods often fall short in providing timely and accurate detection, necessitating more advanced approaches. This project leverages machine learning techniques to develop precise computational models aimed at early detection. By enhancing sensitivity and specificity through ML algorithms, this research aims to revolutionize cancer diagnosis and treatment planning, paving the way for personalized medicine and improved patient outcomes.

## Algorithms Used
### Logistic Regression-Decision Tree
Combines logistic regression for capturing linear relationships with decision trees for handling non-linear patterns effectively.

### K-Nearest Neighbors-Decision Tree
Integrates the instance-based learning of K-Nearest Neighbors with decision trees to enhance classification accuracy and adaptability.

### Naïve Bayes-Random Forest
Leverages Naïve Bayes' probabilistic approach with Random Forest's ensemble learning for robust classification performance.

### Support Vector Machine-Decision Tree
Utilizes Support Vector Machine's ability to find complex decision boundaries with decision trees for comprehensive feature interaction modeling.

### Support Vector Machine-Random Forest
Combines SVM's strength in high-dimensional spaces with Random Forest's resilience to noise for improved generalization.

### Ensemble Model: Random Forest, Gradient Boosting, Extra Trees, and SVC
An ensemble of multiple classifiers including Random Forest, Gradient Boosting, Extra Trees, and Support Vector Classifier (SVC) with Logistic Regression as the final estimator. This approach harnesses the diversity of models to achieve superior predictive accuracy.

## Methodology
### Architecture
The project architecture follows a structured approach beginning with data preprocessing, which includes SMOTE oversampling to address class imbalance, label encoding for categorical variables, and handling missing values. Feature selection is performed using ensemble classifiers like Random Forest, Gradient Boosting, and Extra Trees, supplemented by XGBoost for ranking feature importance. The final model employs a StackingClassifier with multiple base estimators and a Logistic Regression final estimator, validated using cross-validation techniques to ensure robustness and generalizability.

### Dataset
The dataset comprises 3,346 CT image volumes of head and neck cancer collected between 2005 and 2017 from the University Health Network (UHN) in Toronto, Canada. It includes standardized CT images and associated clinical, therapeutic, and demographic data, making it suitable for training prognostic models and quantitative imaging studies.

### Model Training and Evaluation
Models are trained using a 70-30 train-test split with Stratified Shuffle Split cross-validation. Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess model performance across different algorithms. The ensemble model, combining Random Forest, Gradient Boosting, Extra Trees, and SVC, achieves the highest accuracy of 97.29%, underscoring its efficacy in oropharyngeal cancer detection.

## Results
The following table presents the accuracies and performance metrics of various hybrid and ensemble models used:

| Sr. No. | Algorithms Used                                          | Accuracy | Precision | Recall | F1-Score |
|---------|----------------------------------------------------------|----------|-----------|--------|----------|
| 1       | Logistic Regression + Decision Tree                      | 88.99%   | 85.86%    | 83.80% | 84.68%   |
| 2       | K-Nearest Neighbors + Decision Tree                     | 90.18%   | 87.29%    | 87.46% | 87.73%   |
| 3       | Naïve Bayes + Random Forest                              | 89.28%   | 86.24%    | 86.40% | 86.32%   |
| 4       | Support Vector Machine + Decision Tree                   | 89.13%   | 88.49%    | 82.41% | 84.42%   |
| 5       | Support Vector Machine + Random Forest                  | 90.93%   | 89.39%    | 85.03% | 86.67%   |
| 6       | Random Forest, Gradient Boosting, Extra Trees, and SVC   | 97.29%   | 96.84%    | 96.83% | 96.83%   |

These results highlight the effectiveness of ensemble models in achieving high accuracy and robustness in oropharyngeal cancer detection tasks.

## Usage
Clone the repository and follow instructions in `README.md` to run models on your dataset.

## Contributions
Contributions welcome! Fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
