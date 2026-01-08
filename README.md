# Fraud Detection: Statistical Analysis & Predictive Modeling

## 📌 Project Overview
This project develops a machine learning solution to detect fraudulent financial transactions. Using a dataset of over 6.3 million transactions, the goal was to identify patterns of illicit activity—specifically in `TRANSFER` and `CASH_OUT` operations—and provide an actionable prevention plan for financial institutions.

## 📊 Business Problem
Fraudulent transactions represent a tiny fraction of total volume (0.12%) but result in significant financial loss. The objective is to:
1. Identify fraudulent transactions with high sensitivity (Recall).
2. Minimize financial loss by catching "empty-account" patterns.
3. Provide a data-driven infrastructure plan to prevent future fraud.

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
*   **Machine Learning:** XGBoost, Scikit-learn
*   **Evaluation:** Precision-Recall Curve (AUPRC), Confusion Matrix, Classification Report

## 🚀 Key Workflow

### 1. Data Cleaning & Statistical Analysis
*   **Initial Shape:** 6,362,620 rows and 11 columns.
*   **Filtering:** Focused analysis on `TRANSFER` and `CASH_OUT` types, as evidence showed fraud occurs exclusively within these categories.
*   **Missing Values:** Confirmed 0 missing values across the dataset.

### 2. Feature Engineering (The "Signature" of Fraud)
To solve multi-collinearity and capture fraudulent logic, I created **error features** to identify discrepancies between the transaction amount and account balances:
*   `errorBalanceOrig` = `newbalanceOrig` + `amount` - `oldbalanceOrg`
*   `errorBalanceDest` = `oldbalanceDest` + `amount` - `newbalanceDest`
*   **Insight:** Legitimate transactions usually balance out to zero, while fraudulent ones often show large discrepancies because fraudsters attempt to "empty" accounts.

### 3. Handling Class Imbalance
Since fraud represents only **0.12%** of the data, I utilized:
*   **Stratified Sampling:** To maintain fraud ratios in training and validation sets.
*   **XGBoost Scale_pos_weight:** Configured to force the model to prioritize the minority (fraud) class.

## 📈 Model Performance
The model was evaluated primarily on **Recall** (to catch as much fraud as possible) and **AUPRC** (due to the high class imbalance).

| Metric | Score |
| :--- | :--- |
| **Recall (Fraud)** | **98%** |
| **AUPRC** | **0.78** |
| **Accuracy** | **92%** |

### Confusion Matrix Insights:
*   **True Positives:** 1,616 (Fraud correctly identified)
*   **False Negatives:** 27 (Fraud missed)
*   **False Positives:** 41,589 (Legitimate transactions flagged)

## 💡 Key Findings
*   **Top Predictor:** `errorBalanceOrig` is the most significant factor, confirming that ledger discrepancies are the "smoking gun" for fraud.
*   **Transaction Type:** Fraud is heavily concentrated in high-volume `TRANSFER` requests.

## 🛡️ Actionable Prevention Plan
Based on the model's logic, the following infrastructure updates are recommended:
1.  **Real-time Ledger Checks:** Implement a rule-based flag for any transaction where `amount + newbalance != oldbalance`.
2.  **Velocity Monitoring:** Apply a "cool-down" period for accounts receiving a `TRANSFER` and immediately attempting a `CASH_OUT`.
3.  **MFA Triggers:** Mandatory Multi-Factor Authentication for any transaction identified as "High Risk" by the XGBoost model.

## 📁 Repository Structure
```bash
├── Data/
│   └── Fraud.csv                # Dataset (External link recommended due to size)
├── Notebooks/
│   └── Analysis_Modeling.ipynb  # Full source code and visualizations
├── README.md                    # Project documentation
└── Requirements.txt             # Necessary libraries
```

### 📂 Dataset
The dataset is available here:  
[Download Dataset](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)


## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/kamachari/Fraud_Detection_Analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to see the step-by-step analysis and model training.

***

### 📝 Final Note from the Author
*While the model achieves a 98% Recall rate, future iterations will focus on increasing Precision to reduce the number of false alarms and improve the customer experience.*
