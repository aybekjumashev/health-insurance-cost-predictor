import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath='data/raw/insurance.csv'):
    # 1. Load data
    df = pd.read_csv(filepath)
    
    # 2. Encode Binary Variables (Categorical to Numeric)
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    
    # 3. Create Interaction Term (Crucial for this dataset)
    # Since we saw BMI and Smoker have a huge combined effect, we multiply them
    df['bmi_smoker'] = df['bmi'] * df['smoker']
    
    # 4. One-Hot Encode 'region' column
    # This turns 'region' into 3 separate binary columns to avoid assigning arbitrary weight
    df = pd.get_dummies(df, columns=['region'], drop_first=True)
    
    # 5. Separate features (X) and target (y)
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # 6. Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Feature Scaling
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    
    cols = X_train.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('../data/raw/insurance.csv')
    print("Data Preprocessing Successful!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("First 3 rows of X_train:")
    print(X_train.head(3))