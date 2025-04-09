import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# Load Dataset
# ================================
print("📂 Loading dataset...")
dataset = pd.read_csv("HousePricePrediction.csv")
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
dataset = dataset.dropna()

# ================================
# Encode Categorical Variables
# ================================
print("🔤 Encoding categorical variables...")
categorical_cols = [col for col in dataset.columns if dataset[col].dtype == "object"]
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_encoded = pd.DataFrame(OH_encoder.fit_transform(dataset[categorical_cols]))
OH_encoded.columns = OH_encoder.get_feature_names_out(categorical_cols)
OH_encoded.index = dataset.index

# Merge encoded with numerical features
numerical_dataset = dataset.drop(categorical_cols, axis=1)
final_df = pd.concat([numerical_dataset, OH_encoded], axis=1)

# ================================
# Split Features and Target
# ================================
X = final_df.drop('SalePrice', axis=1)
y = final_df['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

# ================================
# Train Random Forest Model
# ================================
print("🌲 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# Evaluate Model
# ================================
y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f"✅ Model Evaluation Complete")
print(f"📉 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# ================================
# Save Model and Encoder
# ================================
model_path = "random_forest_house_price_model.pkl"
encoder_path = "house_price_encoder.pkl"

joblib.dump(model, model_path)
joblib.dump(OH_encoder, encoder_path)

print(f"💾 Model saved to: {model_path}")
print(f"💾 Encoder saved to: {encoder_path}")
