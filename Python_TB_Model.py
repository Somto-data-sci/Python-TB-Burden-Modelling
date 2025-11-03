import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- 1. Load and Prepare Data ---

# Load data
df = pd.read_csv('TB_Burden_Country.csv')

# Define Target and Key Features based on WHO data columns
TARGET_COL = 'Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population'
FEATURES = [
    'Estimated prevalence of TB (all forms) per 100 000 population',
    'Estimated HIV in incident TB (percent)',
    'Case detection rate (all forms), percent'
]

# Select columns and drop rows with missing values after coercing to numeric
df_model = df[[TARGET_COL] + FEATURES].copy()

for col in [TARGET_COL] + FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

df_model.dropna(inplace=True)

# --- 2. Feature Engineering: Create the Classification Target ---

# Use the 75th percentile as a high-risk threshold.
quartile_threshold = df_model[TARGET_COL].quantile(0.75)

# Binary Target: 1 if mortality rate is above the 75th percentile (HIGH RISK)
df_model['High_TB_Risk'] = np.where(df_model[TARGET_COL] >= quartile_threshold, 1, 0)
y = df_model['High_TB_Risk']
X = df_model[FEATURES]

# --- 3. Model Training and Evaluation ---

# Split data (70/30) with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and Train the Logistic Regression Model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Model Evaluation (You will see this output in your PyCharm console)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low Mortality', 'High Mortality'])

# Output Model Coefficients
coefficients = model.coef_[0]

print("--- LOGISTIC REGRESSION MODEL RESULTS (Predicting High TB Mortality Risk) ---")
print(f"Model Accuracy on Test Data: {accuracy:.4f}")
print("\nClassification Report (Key Metrics):")
print(report)

print("\nModel Coefficients (Feature Importance):")
feature_names = X.columns
for name, coef in zip(feature_names, coefficients):
    print(f" - {name}: {coef:.4f}")