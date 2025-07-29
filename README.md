# Employee-Salary-Prediction-Jayaram-Sanku
# Employee Salary Prediction using Machine Learning

## Project Overview

This project focuses on building a machine learning model to predict an individual's income level (whether it's `>50K` or `<=50K` annually) based on demographic and employment-related attributes. The goal is to classify individuals into one of these two income brackets.

## Dataset

The project utilizes the `adult.csv` dataset, which contains various features such as:
- `age`: Age of the individual.
- `workclass`: Type of employer.
- `fnlwgt`: Final weight (sampling weight used by the Census).
- `education`: Highest level of education achieved.
- `educational-num`: Numerical representation of education level.
- `marital-status`: Marital status.
- `occupation`: Occupation category.
- `relationship`: Relationship status.
- `race`: Race of the individual.
- `gender`: Gender of the individual.
- `capital-gain`: Capital gains.
- `capital-loss`: Capital losses.
- `hours-per-week`: Weekly working hours.
- `native-country`: Country of origin.
- `income`: Target variable (`>50K` or `<=50K`).

## Problem Statement

The core problem is to develop a robust binary classification model that accurately predicts an individual's income bracket. This has potential applications in HR for fair compensation analysis, in economic studies to understand income disparities, and in policy-making.

## System Approach

The project follows a standard machine learning pipeline:
1.  **Data Loading:** Loading the `adult.csv` dataset.
2.  **Initial Exploration:** Understanding data structure, types, and initial patterns.
3.  **Data Preprocessing:**
    * Handling missing values (replacing '?' with NaN and imputing with mode for categorical features).
    * Encoding the target variable (`income`) using Label Encoding.
    * Applying One-Hot Encoding to all other categorical features.
4.  **Model Selection:** Choosing the Random Forest Classifier for its robust performance in classification tasks.
5.  **Model Training:** Splitting data into training (80%) and testing (20%) sets, then training the model.
6.  **Model Evaluation:** Assessing performance using metrics like Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Installation & Setup

To run this project, you need Python and the following libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
