# Finding Donors for CharityML

> Udacity — Intro to Machine Learning with PyTorch Nanodegree

## Overview

This project uses supervised learning to predict whether an individual earns more than $50,000 per year, based on data from the 1994 U.S. Census. The goal is to help a fictitious charity, **CharityML**, identify potential donors by accurately modeling income levels from publicly available demographic features.

The dataset originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income) and was featured in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"* by Ron Kohavi and Barry Becker.

## Problem Statement

CharityML needs to efficiently target individuals who are most likely to donate. Predicting whether someone earns above $50K helps the organization decide whom to contact and how much to request. The challenge is a **binary classification** task on a dataset with 45,222 records and 13 features (after preprocessing).

## Approach

1. **Data Exploration** — Analyzed class distribution (~75% earn ≤$50K, ~25% earn >$50K) and feature characteristics.
2. **Preprocessing** — Applied log-transformation to skewed features (`capital-gain`, `capital-loss`), normalized numerical features with `MinMaxScaler`, and one-hot encoded categorical variables, resulting in 103 features.
3. **Model Evaluation** — Benchmarked three supervised learning algorithms against a naive predictor using accuracy and F₀.₅ score:
   - **Random Forest Classifier**
   - **AdaBoost Classifier**
   - **Gaussian Naive Bayes**
4. **Model Selection & Optimization** — Selected **AdaBoost** as the best candidate and tuned its hyperparameters with `GridSearchCV`.
5. **Feature Importance** — Extracted the top 5 most predictive features and evaluated model performance on the reduced feature set.

## Results

| Metric         | Naive Predictor | Unoptimized AdaBoost | Optimized AdaBoost |
|:--------------:|:---------------:|:--------------------:|:------------------:|
| Accuracy       | 0.2478          | 0.8638               | 0.8706             |
| F₀.₅ Score     | 0.2917          | 0.7333               | 0.7447             |

The top 5 features by importance were: **age**, **hours-per-week**, **capital-gain**, **capital-loss**, and **education-num**.

## Repository Structure

```
├── finding_donors.ipynb    # Main Jupyter Notebook with full analysis
├── finding_donors.html     # HTML export of the notebook
├── visuals.py              # Helper visualization functions
├── census.csv              # Dataset
├── project_description.md  # Original project brief
└── README.md               # This file
```

## Tech Stack

- **Python 3.6**
- **NumPy**, **Pandas** — data manipulation
- **Matplotlib** — visualization
- **scikit-learn** — preprocessing, model training, evaluation, and tuning

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install numpy pandas matplotlib scikit-learn jupyter`
3. Launch the notebook: `jupyter notebook finding_donors.ipynb`

## License

This project was completed as part of the Udacity Machine Learning Nanodegree program.
