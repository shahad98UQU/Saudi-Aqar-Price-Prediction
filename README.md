# Saudi Arabia Real Estate Price Prediction Framework

## Project Overview
This repository contains an end-to-end data science and machine learning pipeline developed to forecast property prices across Saudi Arabia using the localized `SA_Aqar` dataset. The framework implements automated data engineering wrappers combined with optimized regression baselines.

---

## Technical Workflow & Features

### 1. Robust Data Cleaning
* **Outlier Mitigation:** Utilizes percentile-based quantiles filtering to eliminate extreme price and structural area anomalies.
* **Smart Imputation:** Automates missing value resolution by filling numeric features via column medians and categorical dimensions via structural modes.

### 2. Feature Engineering Pipeline
* Seamlessly bundles data scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) into unified Scikit-Learn `Pipeline` workflows to prevent data leakage during training.

### 3. Model Evaluation
* Trains and cross-evaluates multiple supervised regression architectures including **Linear Regression** and **Ensemble Methods**.
* Computes standard performance metrics including **$R^2$ (Coefficient of Determination)** and **Mean Absolute Error (MAE)** to analyze model convergence and prediction fidelity.

---

## Project Structure
* `saudi_aqar_price_prediction.py`: Full source code including preprocessing modules, visual analytics mapping, and model benchmarking pipelines.

## Tools & Frameworks
* **Core Libraries:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Python 3
