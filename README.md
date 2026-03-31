# 🏭 Bankruptcy Prediction using Advanced ML Ensemble  
**End-Semester Project | Supply Chain Finance Risk Modeling**

---

## 📌 Overview
This project focuses on predicting company bankruptcy using advanced machine learning techniques on the **Polish Companies Bankruptcy Dataset**.

It introduces a **3-level stacking ensemble**, **adaptive feature selection**, and **financial domain-based feature engineering** to significantly improve prediction performance under severe class imbalance.

---

## 🚀 Key Features

### 🔹 1. Adaptive Feature Selection (SHAP-based)
- Per-year dynamic feature selection
- Uses SHAP values (handles 3D outputs correctly)
- Selects top-k most important features for each year

### 🔹 2. Composite Financial Indicators (12 Features)
Derived from financial distress models:
- Altman Z-score  
- Zmijewski model  
- Springate model  
- Grover score  
- Taffler model  
- Cash Conversion Cycle (CCC)  
- Debt Service Score (DSS)  
- Profit Collapse Indicator (PCI)  
- Liquidity Stress Index (LSI)  
- Distress Count Score (CDS)  
- SCF Finance Need Index (SCFNI)  
- Leverage Danger Zone (LDZ)

---

### 🔹 3. 3-Level Stacking Ensemble

Level 0: XGBoost + Random Forest + LightGBM + Logistic Regression + Gradient Boosting  
↓  
Level 1: Meta Learner (Logistic Regression / XGBoost)  
↓  
Level 2: Threshold Optimization (F1-based)

---

### 🔹 4. Class Imbalance Handling
- SMOTE (Synthetic Minority Oversampling)
- Cost-sensitive learning
- F1-score optimized decision threshold

---

### 🔹 5. Robust Data Pipeline
- Median Imputation
- Robust Scaling
- Outlier Clipping (±5σ)
- Leakage-safe train-test processing

---

### 🔹 6. Risk Ranking System
Generates **Top-50 high-risk companies per year** categorized as:
- 🔴 CRITICAL  
- 🟠 HIGH  
- 🟡 MEDIUM  
- 🔵 WATCH  

---

### 🔹 7. Checkpointing System
- Saves progress year-wise
- Enables resume for long training runs

---

## 📊 Dataset

- **Source:** Polish Companies Bankruptcy Dataset  
- **Years Covered:** 5  
- **Total Companies:** 43,405  
- **Bankrupt Cases:** 2,091  
- **Imbalance Ratio:** Up to 25:1  

---

## 📈 Exploratory Analysis

Includes:
- Class imbalance visualization  
- Missing value analysis  
- Feature importance (Mann-Whitney U test)  
- Correlation heatmaps  
- Spatial separation metrics  

---

## 📐 Key Insight

Adding composite financial features improved class separation significantly:

- **Euclidean Distance:** +128%  
- **Fisher Discriminant Ratio:** +30%  

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- XGBoost  
- LightGBM  
- SHAP  
- Optuna  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Imbalanced-learn  

---

## ⚙️ Installation

```bash
pip install imbalanced-learn xgboost lightgbm scikit-learn matplotlib seaborn pandas numpy scipy joblib optuna shap
```

---

## ▶️ How to Run

1. Upload dataset ZIP:
```
polish_companies_bankruptcy_data.zip
```

2. Open the notebook:
```
Adaptive Risk Prediction in SCF.ipynb
```

3. Run all cells sequentially

---

## 📁 Project Structure

```
├── Adaptive Risk Prediction in SCF.ipynb
├── data/
│   └── polish_companies_bankruptcy_data.zip
└── README.md
```

---

## 📌 Applications

- Credit risk assessment  
- Supply chain finance monitoring  
- Banking & lending decisions  
- Corporate risk analytics  

---

## 📚 References

- Zieba et al. (2016)  
- Barboza et al. (2017)  

---

## ⭐ Future Improvements

- Real-time bankruptcy prediction API  
- Deployment using Flask / FastAPI  
- Integration with financial dashboards  
- Deep learning-based hybrid models  
