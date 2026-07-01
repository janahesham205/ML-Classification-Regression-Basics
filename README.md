# Machine Learning Foundations: Classification & Regression

This repository contains implementation of core Machine Learning tasks as part of the **Introduction to Machine Learning** course at Alexandria National University. The project demonstrates proficiency in data preprocessing, handling class imbalance, model evaluation, and hyperparameter tuning using standard frameworks like NumPy, Pandas, and Scikit-Learn.

## 📌 Project Overview
The project is divided into two major problem statements:
1. **Binary Classification:** Predicting particle types using the MAGIC Gamma Telescope dataset with **K-Nearest Neighbors (K-NN)**.
2. **Regression:** Predicting housing prices using the California Housing dataset with **Linear, Lasso, and Ridge Regression** variants.

---

## 🔬 Problem Statement 1: Particle Classification (MAGIC Gamma Telescope)

### 📊 Dataset Description
The dataset simulates registration of high-energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope. 
* **Classes:** Gammas (Signal: 12,332 events) and Hadrons (Background: 6,688 events).
* **Challenge:** Highly imbalanced classes.

### 🛠️ Methodology & Implementation
1. **Handling Class Imbalance:** Randomly downsampled the majority class (`gamma`) to match the size of the minority class (`hadron`), ensuring a balanced 50:50 distribution.
2. **Data Splitting:** Partitioned the dataset randomly into **70% Training**, **15% Validation** (for hyperparameter tuning), and **15% Testing** (held out for final evaluation).
3. **Model:** Applied the **K-NN Classifier** and experimented with various $k$ values to find the optimal hyperparameter.
4. **Evaluation Metrics:** Evaluated models using Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.

---

## 🏠 Problem Statement 2: Housing Price Prediction (California Census)

### 📊 Dataset Description
Based on the 1990 California census, this dataset contains geographic, demographic, and economic features of various districts used to predict the **Median House Value** (Target variable).
* **Key Features:** Median Income, Median Age, Total Rooms/Bedrooms, Population, Households, Coordinates, and Distances to major coastal cities (LA, San Diego, San Jose, San Francisco).

### 🛠️ Methodology & Implementation
1. **Data Splitting:** Followed a strict **70/15/15** split strategy for training, validation, and testing sets.
2. **Models Applied:** Evaluated and compared three linear baseline models:
   * **Linear Regression** (Standard Ordinary Least Squares)
   * **Lasso Regression** (L1 Regularization for feature selection)
   * **Ridge Regression** (L2 Regularization to prevent overfitting)
3. **Evaluation Metrics:** Evaluated model performance using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

---

## 🛠️ Tech Stack & Tools Used
* **Language:** Python
* **Libraries:** NumPy, Pandas, Scikit-Learn
* **Environment:** Jupyter Notebook

## 📈 Key Learnings
* Gained hands-on experience in addressing class imbalance through data resampling.
* Mastered the strict separation of Training/Validation/Testing data to prevent data leakage during hyperparameter tuning.
* Explored the effect of regularization parameters ($L1$ and $L2$) in stabilizing linear models against overfitting.
