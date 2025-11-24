# vehicle-insurance-fraud-detection
Machine learning project detecting fraudulent vehicle insurance claims, with EDA, feature engineering, and a fraud prediction model.
# Vehicle Insurance Fraud Detection

This project focuses on detecting **fraudulent vehicle insurance claims** using machine learning.

## Project Goals

- Analyze historical claim data to understand the **proportion and patterns of fraud**.  
- Build a **predictive model** that can flag potentially fraudulent claims.  
- Focus on reducing **Type II errors** (missing actual fraud cases), even if it allows a moderate level of Type I errors (flagging some genuine claims as suspicious).  

## Dataset

- Source: Vehicle insurance claims dataset (`carclaims.csv`).
- Contains information about policyholders, claim details, vehicle attributes, and a target column indicating whether **fraud was found**.

> Note: The original dataset may not be included in this repository. If not, you can replace this section with instructions on how to obtain it.

## Key Steps

1. **Exploratory Data Analysis (EDA)**
   - Checked data shape, missing values, and duplicates.
   - Compared distributions between **fraud** and **non-fraud** cases separately to identify important patterns.

2. **Feature Engineering & Transformation**
   - Grouped months, weeks, and days into meaningful categories.
   - Combined rare vehicle makes into buckets.
   - Binned variables like vehicle price, age of vehicle, and age of policyholder into risk-based groups.
   - Removed weak or highly unbalanced features that didn’t add value to the model.

3. **Encoding & Preparation**
   - Converted categorical variables into numeric using one-hot encoding.
   - Split data into features (`X`) and target (`y`).
   - Handled class imbalance (fraud vs non-fraud) using appropriate techniques before training.

4. **Modeling**
   - Tried multiple machine learning models:
     - Decision Tree
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - LightGBM
     - Stacking Classifier (ensemble)
   - Used **K-Fold cross-validation** to evaluate performance.

5. **Evaluation**
   - Focused on metrics like:
     - **F1-score**
     - **Recall** (especially for fraud class)
     - **Precision**
     - **Accuracy**
   - Tuned hyperparameters for XGBoost using `GridSearchCV` and compared results.

## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy`  
  - `matplotlib`, `seaborn`  
  - `scikit-learn`  
  - `xgboost`, `lightgbm`
