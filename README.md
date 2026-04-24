# Health Insurance Cost Regression Analysis 🏥

This project predicts individual health insurance costs based on demographic and health indicators using Machine Learning. It features a multiple regression analysis and a user-facing premium estimator built with Streamlit.

## 📊 Project Overview
The goal of this project is to understand the key drivers of medical insurance costs and provide an accurate estimation tool. By analyzing the interaction between smoking status and Body Mass Index (BMI), we developed a model that explains approximately **86.5%** of the variance in medical charges.

## 💡 Key Business Insights
- **The Smoker Penalty:** Smoking is the single most significant factor. On average, smokers pay ~300% more than non-smokers.
- **Interaction Effect:** The combination of smoking and high BMI (>30) leads to a drastic exponential increase in costs.
- **Age Factor:** Medical costs increase steadily by approximately $250-$300 for every additional year of age.

## 🛠️ Tech Stack
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn.
- **Machine Learning:** Scikit-learn (RandomForest, StandardScaler), Statsmodels (OLS Regression).
- **Web App:** Streamlit.
- **Deployment-ready:** Joblib for model serialization.

## 📈 Model Performance
We evaluated two models: Linear Regression and Random Forest.
- **Random Forest (Best Model):**
  - **MAE:** ~$2,538 (Mean Absolute Error)
  - **R² Score:** 0.8654
- The model successfully meets the MVP requirement of an error range under $3,000.

## 🚀 How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python src/model.py
   ```
3. **Launch the Web App:**
   ```bash
   streamlit run app/main.py
   ```

## 📁 Repository Structure
- `data/`: Contains the raw CSV dataset.
- `notebooks/`: Exploratory Data Analysis (EDA) and visualization.
- `src/`: Source code for data preprocessing and model training.
- `models/`: Saved model and scaler files (.pkl).
- `app/`: Streamlit application code.