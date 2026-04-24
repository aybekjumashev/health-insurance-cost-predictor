import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from data_prep import load_and_preprocess_data

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2:   {r2:.4f}\n")
    return mae, rmse, r2

def train_and_evaluate():
    # 1. Load Data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('../data/raw/insurance.csv')
    
    # 2. Linear Regression (scikit-learn)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    evaluate_model(y_test, lr_preds, "Linear Regression")

    # 3. Statsmodels OLS for Statistical Summary
    # Adding a constant for the intercept
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    print("--- OLS Regression Summary (Statsmodels) ---")
    print(ols_model.summary().tables[1]) # Only print coefficients table
    print("\n")

    # 4. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    evaluate_model(y_test, rf_preds, "Random Forest")

    # 5. Save the best model (Random Forest) and the scaler for the Streamlit app
    os.makedirs('../models', exist_ok=True)
    joblib.dump(rf, '../models/rf_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("Models and scaler saved successfully to '../models/' folder.")

    # 6. Visualization: Predicted vs Actual (Random Forest)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=rf_preds, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Charges (Random Forest)")
    plt.xlabel("Actual Charges ($)")
    plt.ylabel("Predicted Charges ($)")

    # 7. Visualization: Residuals Analysis
    residuals = y_test - rf_preds
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('../images', exist_ok=True)
    plt.savefig('../images/model_evaluation.png')
    print("Evaluation plots saved to '../images/model_evaluation.png'.")
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()