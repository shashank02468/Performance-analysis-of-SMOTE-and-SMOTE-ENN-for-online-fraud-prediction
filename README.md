# Performance-analysis-of-SMOTE-and-SMOTE-ENN-for-online-fraud-prediction
## Project Overview
This project will explore two popular resampling techniques one is SMOTE which is Synthetic Minority Oversampling Technique and the other one is SMOTE combined with Edited Nearest Neighbour (SMOTE-ENN) technique. This is mainly used for prediction of online fraud for highly imbalanced transaction data. The core idea is to compare these two methods in terms of accuracy, training time and the overall efficiency when applied to different Machine Learning models.
### Data Source
The dataset employed here is obtained from  git hub repository [https://github.com/Shamimansari42/Online-Payments-Fraud-Detection-Dataset]
## 🕵️‍♂️ Research Question
How does the online fraud prediction performance differ between models trained using the **SMOTE** oversampling technique and the **SMOTE-ENN** (Edited Nearest Neighbors) hybrid sampling technique when evaluated on imbalanced, real-time transaction data?
## 🚀 Aim
To evaluate and compare the effectiveness of **SMOTE** and **SMOTE-ENN** resampling techniques in improving the performance of machine learning models for online fraud detection using imbalanced transaction datasets.

## 📋 Objectives
- Explore and preprocess the Online Payments Fraud Detection dataset
for quality and suitability.  
- Implement ML models: Logistic Regression (LR), Random Forest (RF),
K-Nearest Neighbor (KNN), and Decision Tree (DT).  
- Apply **SMOTE** oversampling and evaluate fraud detection
performance.  
- Apply **SMOTE-ENN** hybrid sampling and analyze impact on
classification and noise reduction.  
- Compare models using accuracy, precision, recall, F1-score, and
confusion matrix.  
- Evaluate computational efficiency and real-world feasibility for
online fraud detection.
## 🛠️ Tools Required
- Python  
- Jupyter Notebook  
- Visual Studio Code
  ## 📦 Packages Used
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `imbalanced-learn`  
- `matplotlib`  
- `seaborn`  
- `time`
## 📊 Dataset Overview
| Feature Name       | Description |
|-------------------|-------------|
| step               | Time step in hours (1 step = 1 hour) |
| type               | Type of transaction (CASH_OUT, PAYMENT, etc.) |
| nameOrig           | Customer initiating the transaction |
| amount             | Transaction amount |
| newbalanceOrig     | Origin account balance after transaction |
| oldbalanceOrg      | Origin account balance before transaction |
| nameDest           | Recipient of transaction |
| newbalanceDest     | Recipient balance after transaction |
| oldbalanceDest     | Recipient balance before transaction |
| isFlaggedFraud     | 1 if flagged, else 0 |
| isFraud            | 1 if transaction is fraudulent, else 0 |
### Methodology
Loaded the dataset using pandas which has 6.3M rows in that 8213 are fraud transactions and 6.3M non fraud which is a highly imbalanced data. Downsampled the data for hyperparameter computational feasibility which now has 1.9M rows in that 1.9M are non fraud and 2464 are fraud.
#### 1.Preprocessing: 
Checked for unique and null values and duplicate rows. Used label encoder for categorical values and performed EDA.
#### 2.Handling Imbalance: 
Applied SMOTE and SMOTE-ENN to balance the class and split the dataset for training testing and validation.
#### 3.Split:
Split the dataset for training testing and validation in which 60% Train, 20% Validation, 20% Test 
#### 4.Model Implementation: 
Implemented Logistic Regression, Random forest, Decision Tree and KNN and used GridSearchCV to find the best parameters.
#### 5.Metrics:
Used Metrics such as precision, recall, f1 score, accuracy and confusion matrix to understand which model is performing better.
## 📈 Results and Analysis
### Validation Results
| Oversampling | Model | Precision | Recall | F1-Score | Accuracy|
|--------------|-------|-----------|--------|----------|---------|
| SMOTE        | LR    | 0.96      | 0.96   | 0.96     | 0.96    |
|              | RF    | 0.99      | 0.99   | 0.99     | 0.99    |
|              | DT    | 0.97      | 0.97   | 0.97     | 0.97    |
|              | KNN   | 0.98      | 0.98   | 0.98     | 0.98    |
| SMOTE-ENN    | LR    | 0.96      | 0.96   | 0.96     | 0.96    |
|              | RF    | 0.98      | 0.98   | 0.98     | 0.98    |
|              | DT    | 0.97      | 0.97   | 0.97     | 0.97    |
|              | KNN   | 0.99      | 0.99   | 0.99     | 0.99    |
### Testing Results
| Oversampling | Model | Precision | Recall | F1-Score | Accuracy|
|--------------|-------|-----------|--------|----------|---------|
| SMOTE        | LR    | 0.96      | 0.96   | 0.96     | 0.96    |
|              | RF    | 0.99      | 0.99   | 0.99     | 0.99    |
|              | DT    | 0.97      | 0.97   | 0.97     | 0.97    |
|              | KNN   | 0.98      | 0.98   | 0.98     | 0.98    |
| SMOTE-ENN    | LR    | 0.96      | 0.96   | 0.96     | 0.96    |
|              | RF    | 0.99      | 0.98   | 0.98     | 0.98    |
|              | DT    | 0.97      | 0.97   | 0.97     | 0.97    |
|              | KNN   | 0.98      | 0.98   | 0.98     | 0.98    |
### Training Results
| Oversampling | Model | Training Accuracy | Training Time (sec) |
|--------------|-------|-----------------|-------------------|
| SMOTE        | LR    | 0.9589          | 395.31            |
|              | RF    | 0.9882          | 264.15            |
|              | DT    | 0.9681          | 7.07              |
|              | KNN   | 1.0000          | 0.10              |
| SMOTE-ENN    | LR    | 0.9608          | 1300.48           |
|              | RF    | 0.9849          | 344.60            |
|              | DT    | 0.9723          | 8.72              |
|              | KNN   | 1.0000          | 0.15              |
---
## 📝 Conclusion
- Both **SMOTE** and **SMOTE-ENN** effectively address
class imbalance.  
- **Random Forest** and **KNN** provide highest
predictive performance (0.98--0.99).  
- **SMOTE-ENN** does not improve accuracy significantly but
increases training time.  
- **SMOTE** offers a faster, computationally efficient solution
without compromising predictive power.  
---
## 🔮 Future Work
- Explore advanced resampling methods like **ADASYN** or
**Borderline-SMOTE**.  
- Investigate ensemble learning algorithms such as **XGBoost**
or **LightGBM** in combination with resampling.  
- Deploy best-performing models in a real-time streaming environment to
test efficiency and response performance.



