# Fraud Detection System

## Overview

This project implements a **fraud detection system** for financial transactions using **XGBoost**. The system can detect fraudulent transactions, flag suspicious transfers, and provide insights into transaction patterns.

## Features

* Train an XGBoost model on historical transaction data.
* Test the model on new transaction data (without fraud labels).
* Identify:

  * High-confidence fraud transactions (>90% accuracy)
  * Flagged fraud transactions (<90% accuracy)
  * Flagged but non-fraud transactions (<90% accuracy)
* Handles large datasets using **split ZIP files** for GitHub storage.

## Data

* Training dataset: fraud_data_split.zip 1-8
* New data for testing: new_transactions.csv
* Data columns:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

 **Results**

* Outputs accuracy and recall on new data
* Lists transaction IDs for:

  * Fraud detected (>90% confidence)
  * Flagged fraud (<90% confidence)
  * Flagged non-fraud (<90% confidence)

## Notes

* Large dataset files are split into multiple ZIPs for GitHub storage.
* Requires Python libraries: pandas, xgboost, scikit-learn.
