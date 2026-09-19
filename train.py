"""
Trains the Linear Regression car price model exactly the way the original
notebook does, then saves everything the Streamlit app needs:
  - linear_regression_model.pkl  -> the trained model
  - label_encoders.pkl           -> one LabelEncoder per categorical column
  - feature_columns.pkl          -> exact column order the model expects
  - metrics.pkl                  -> R2 / RMSE for Linear, Ridge, Lasso (for the UI)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# ---- Step 1: Load data -----------------------------------------------------
df = pd.read_csv("car_data.csv")

# ---- Step 2: Handle missing values -----------------------------------------
df = df.dropna(subset=["Price", "EngineV"])

# ---- Step 3: Remove EngineV outliers (IQR method) --------------------------
Q1 = df["EngineV"].quantile(0.25)
Q3 = df["EngineV"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df["EngineV"] >= lower) & (df["EngineV"] <= upper)]

# ---- Step 4: Log-transform price, drop raw Price + Model -------------------
df["log_price"] = np.log(df["Price"])
df = df.drop(columns=["Price", "Model"])  # Model has 312 uniques -> dropped

# ---- Step 5: Label-encode categorical columns -------------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---- Step 6: Train/test split ------------------------------------------------
X = df.drop(columns=["log_price"])
y = df["log_price"]
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Step 7: Train all three models & collect metrics -----------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
}

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics[name] = {
        "r2": r2_score(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
    }

# Final model to ship = Linear Regression (matches the notebook's choice)
final_model = models["Linear Regression"]

# ---- Step 8: Save everything -------------------------------------------------
joblib.dump(final_model, "linear_regression_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(metrics, "metrics.pkl")

print("Feature columns:", feature_columns)
print()
for name, m in metrics.items():
    print(f"{name}: R2 = {m['r2']:.4f}  RMSE = {m['rmse']:.4f}")
print("\nSaved: linear_regression_model.pkl, label_encoders.pkl, feature_columns.pkl, metrics.pkl")
