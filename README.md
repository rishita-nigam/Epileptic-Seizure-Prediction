# 🧠 EEG-Based Seizure Detection using Machine Learning

Automated seizure detection from EEG signals using classical ML and 
signal processing on publicly available epilepsy datasets.

---

## Overview

Epileptic seizures are characterized by abnormal electrical activity in 
the brain. This project builds a machine learning pipeline to classify 
EEG segments as seizure or non-seizure, enabling automated, real-time 
detection without manual expert review.

---

## Dataset

- **Source**: UCI Epileptic Seizure Recognition Dataset  
- 11,500 EEG samples × 178 time-series features  
- Binary classification: Seizure (1) vs Non-Seizure (0)  
- Balanced classes after preprocessing

---

## Pipeline

| Stage | Details |
|-------|---------|
| Preprocessing | Normalization, class balancing |
| Features | Raw EEG time-steps + statistical aggregates |
| Models | Random Forest, SVM, Logistic Regression |
| Evaluation | Accuracy, F1, Confusion Matrix, ROC-AUC |

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | ~95%+ | — |
| SVM | — | — |

*(Fill in your actual numbers)*

---

## How to Run

```bash
git clone https://github.com/rishita-nigam/seizure-detection
cd seizure-detection
pip install -r requirements.txt
python seizure_detection.py
```

---

## Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn

---

## Tech Stack

Python · scikit-learn · NumPy · pandas · Matplotlib

---

## B.Tech EEE — VIT Vellore
