# Global Health Predictive Modeling: Tuberculosis Mortality Risk (Python Project)

## 1. Project Objective and Contribution

This project applies Logistic Regression to WHO data to build a predictive model for **Tuberculosis (TB) Mortality Risk**.

The goal is to quantify the factors (features) driving high TB death rates to provide data-driven recommendations for **resource allocation and pharmacoeconomic planning.**

## 2. Key Model Results and Policy Insight

The model successfully classified countries with **94.82% Accuracy** on the test set.

### Primary Policy Insight: Highest Return on Investment (ROI)

The model coefficients reveal the most effective intervention for reducing mortality risk:

* **Case detection rate (all forms), percent: -0.0473 (Strongest Negative)**
    * **Interpretation:** This is the core finding. The negative value confirms that **increasing the efficiency of diagnostic systems** (finding cases faster) is the single most powerful strategy for lowering mortality risk. This supports investment in diagnostics and surveillance.

* **Estimated HIV in incident TB (percent): 0.0114 (Positive)**
    * **Interpretation:** Confirms that HIV co-infection significantly increases mortality risk and requires high-intensity treatment resources.

* **Estimated prevalence of TB: 0.0100 (Positive)**
    * **Interpretation:** The existing size of the TB epidemic slightly increases mortality risk.

### Classification Metrics Summary

* Accuracy on Test Data: 0.9482
* F1-Score (High Mortality Class): 0.90

## 3. Technical Details

* **Model:** Logistic Regression (chosen for feature interpretability).
* **Data Source:** World Health Organization (WHO) Tuberculosis Burden by Country.
* **Feature Engineering:** The continuous Mortality Rate was converted into a binary target: 1 (High Risk) vs. 0 (Low Risk) using the 75th percentile threshold.

---