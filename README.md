# Bank Term Deposit Subscription Prediction & Customer Segmentation

An end-to-end machine learning project analyzing demographic, financial, and behavioral data to predict customer term deposit subscriptions and uncover key customer personas.

---

## Executive Summary

Predicting bank term deposit subscriptions presents a significant challenge due to severe class imbalance (only ~7% of prospective customers subscribe). This project evaluates Decision Trees, class weighting strategies, and supplementary behavioral features to maximize subscription detection ($F_1$-score). 

Additionally, unsupervised learning techniques (**K-Means Clustering** and **Principal Component Analysis**) were applied to segment subscribing customers into actionable demographic cohorts.

---

## Key Results & Findings

* **Optimal Model:** A Decision Tree Classifier trained strictly on demographic and financial features achieved the best predictive balance ($F_1\text{-score} = 0.42$ for subscribers) when using a **31:1 class weight ratio** (Subscribers : Non-Subscribers).
* **Feature Impact:** 
  * **Behavioral Features:** Adding behavioral attributes (e.g., call duration, contact frequency, date) via a Nearest Centroid refinement step did not meaningfully improve predictive accuracy.
  * **Missing Value Imputation:** Predicting missing education values via secondary ML modeling slightly reduced downstream predictive power compared to handling raw features directly.
* **Customer Cohorts:** Unsupervised clustering revealed that subscriber profiles are primarily driven by **Age** and **Account Balance**.

---

## Methodology & Machine Learning Pipeline

Raw Data (40,000 Records)
└── Demographic & Financial Features
├── Optuna Tuning & Class Weighting (31:1) ──> Decision Tree Classifier (F1: 0.42)
└── Unsupervised Analysis
├── K-Means Clustering (k=5) ─────────────> Demographic Cohort Generation
└── Principal Component Analysis (2D) ────> Dimensionality Reduction & Visualization

### 1. Supervised Learning & Optimization
* **Baseline Challenge:** Class imbalance (93% non-subscribers vs. 7% subscribers).
* **Metric Choice:** $F_1$-score on the positive subscriber class to balance recall and precision without assuming fixed misclassification costs.
* **Hyperparameter Tuning:** Kept class weights fixed during individual search passes using **Optuna** to systematically evaluate model hyperparameter spaces across manual weight configurations.

### 2. Feature Engineering Experiments
* **Education Imputation:** Noticed that missing education values (~3% of dataset) corresponded chronologically to the earliest contacts. Trained a Decision Tree to impute education based on demographic predictors; however, feeding imputed values back into the primary model slightly degraded test performance.
* **Behavioral Refinement:** Applied the **Nearest Centroid** algorithm on behavioral contact metrics to refine predictions from the demographic stage. Behavioral variables provided minimal gain and were ultimately excluded for model simplicity.

---

## Customer Segmentation (Unsupervised Analysis)

To better understand *who* subscribes, **K-Means Clustering** was run on subscriber data to build 5 distinct customer cohorts, followed by **Principal Component Analysis (PCA)** for 2D visualization.

### Cohort Summary

| Cohort | Name / Profile | Mean Age | Mean Balance ($) | Dominant Occupation | Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Senior Citizens | 63 | $2,318 | Retired | Highest average account balance |
| **1** | Young Adult | 27 | $1,658 | Student | Youngest demographic group |
| **2** | Low Balance | 42 | $1,400 | Mixed | Lowest average account balance |
| **3** | Mid-Career | 33 | $1,645 | Mixed | Second-youngest cohort |
| **4** | Management | 43 | $1,757 | Management | Mid-career professionals |

### PCA Visual Insights
* **PC1 (Horizontal Axis):** Strongly correlates with **Age**, anchoring the Senior cohort (Cohort 0) and Young Adult cohort (Cohort 1) at opposite poles.
* **PC2 (Vertical Axis):** Driven primarily by **Account Balance**, cleanly separating low-balance subscribers from higher-earning groups.

---

## Technologies Used

* **Python 3.x**
* **scikit-learn** (Decision Trees, Nearest Centroid, K-Means, PCA)
* **Optuna** (Hyperparameter Optimization)
* **Pandas / NumPy** (Data Preprocessing)
* **Matplotlib / Seaborn** (Data Visualization)

---

## Repository Structure

```text
├── data/               # Dataset storage (raw and processed)
├── notebooks/          # Jupyter notebooks for EDA, modeling, and PCA visualizations
├── src/                # Python code for data pipelines and modeling
├── README.md           # Project documentation
└── requirements.txt    # Required dependencies
