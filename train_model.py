import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
start = time.time()

# --------------------------------------------------
# 1. LOAD TRANSACTION DATA
# --------------------------------------------------
if os.path.exists("cleaned_retail_data.csv"):
    df = pd.read_csv("cleaned_retail_data.csv")
    print("Loaded cleaned_retail_data.csv")
elif os.path.exists("Online Retail.xlsx"):
    df = pd.read_excel("Online Retail.xlsx")
    print("Loaded Online Retail.xlsx")
else:
    raise FileNotFoundError(
        "Put cleaned_retail_data.csv or Online Retail.xlsx "
        "in the same folder as train_model.py."
    )

# Normalize column names
df.columns = [str(c).strip() for c in df.columns]

required = [
    "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"
]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. Available: {list(df.columns)}"
    )

# --------------------------------------------------
# 2. CLEAN DATA
# --------------------------------------------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

df = df.dropna(
    subset=["InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]
)
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

if "InvoiceNo" in df.columns:
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ].copy()

df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df = df.sort_values("InvoiceDate")

if df.empty:
    raise ValueError("No valid transaction rows remain after cleaning.")

print("Clean transaction rows:", len(df))
print("Unique customers:", df["CustomerID"].nunique())

# --------------------------------------------------
# 3. DEFINE HISTORY AND FUTURE PERIODS
# Use the first 80% of the observed time span for history.
# The remaining 20% is the future prediction period.
# --------------------------------------------------
min_date = df["InvoiceDate"].min().normalize()
max_date = df["InvoiceDate"].max().normalize()
total_days = (max_date - min_date).days

if total_days < 10:
    raise ValueError(
        "Not enough date coverage to create a future-spending target."
    )

cutoff = min_date + pd.Timedelta(days=int(total_days * 0.8))
future_end = max_date + pd.Timedelta(days=1)

history = df[df["InvoiceDate"] < cutoff].copy()
future = df[
    (df["InvoiceDate"] >= cutoff) &
    (df["InvoiceDate"] < future_end)
].copy()

if history.empty or future.empty:
    raise ValueError(
        "History or future period is empty. Check transaction dates."
    )

print("History cutoff:", cutoff.date())
print("Future period ends:", max_date.date())

# --------------------------------------------------
# 4. BUILD CUSTOMER FEATURES FROM HISTORY ONLY
# --------------------------------------------------
reference_date = cutoff

history["InvoiceNo"] = (
    history["InvoiceNo"].astype(str)
    if "InvoiceNo" in history.columns
    else history.index.astype(str)
)

features = history.groupby("CustomerID").agg(
    Total_Spending=("Revenue", "sum"),
    Purchase_Frequency=("InvoiceNo", "nunique"),
    Total_Quantity=("Quantity", "sum"),
    Average_Order_Value=("Revenue", "mean"),
    First_Purchase=("InvoiceDate", "min"),
    Last_Purchase=("InvoiceDate", "max")
).reset_index()

features["Days_Since_Last_Purchase"] = (
    reference_date - features["Last_Purchase"]
).dt.days.clip(lower=0)

features["Customer_Tenure_Days"] = (
    features["Last_Purchase"] - features["First_Purchase"]
).dt.days.clip(lower=0)

# Store readable customer feature table
features.to_csv("customer_features.csv", index=False)

# --------------------------------------------------
# 5. CREATE THE FUTURE-SPENDING TARGET
# --------------------------------------------------
target = future.groupby("CustomerID")["Revenue"].sum()
target.name = "FutureSpending"

data = features.merge(
    target,
    on="CustomerID",
    how="left"
)

# Customers with no purchases in the future period get zero spending
data["FutureSpending"] = data["FutureSpending"].fillna(0)

# Save results used by the dashboard
data.to_csv("customer_ml_results.csv", index=False)

# --------------------------------------------------
# 6. PREPARE MODEL INPUTS
# --------------------------------------------------
feature_cols = [
    "Total_Spending",
    "Purchase_Frequency",
    "Total_Quantity",
    "Average_Order_Value",
    "Days_Since_Last_Purchase",
    "Customer_Tenure_Days"
]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
y = data["FutureSpending"].clip(lower=0)

# Chronological split already separates the future target.
# Hold out customers for evaluation.
if len(data) < 10:
    raise ValueError("Too few customers to train and evaluate the model.")

rng = np.random.RandomState(42)
indices = rng.permutation(len(data))
split_at = max(1, int(len(data) * 0.8))
train_idx = indices[:split_at]
test_idx = indices[split_at:]

if len(test_idx) == 0:
    raise ValueError("Not enough customers for a test set.")

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# --------------------------------------------------
# 7. TRAIN A SMALLER RANDOM FOREST
# --------------------------------------------------
model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    func=np.log1p,
    inverse_func=np.expm1
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------
predictions = np.maximum(model.predict(X_test), 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

pd.DataFrame({
    "ActualSpending": y_test.to_numpy(),
    "PredictedSpending": predictions
}).to_csv("model_predictions.csv", index=False)

rf = model.regressor_.named_steps["rf"]

pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf.feature_importances_
}).sort_values(
    "Importance", ascending=False
).to_csv("feature_importance.csv", index=False)

pd.DataFrame([{
    "Model": "Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv("model_comparison.csv", index=False)

# --------------------------------------------------
# 9. CUSTOMER SEGMENTATION
# --------------------------------------------------
cluster_cols = [
    "Total_Spending",
    "Purchase_Frequency",
    "Average_Order_Value",
    "Days_Since_Last_Purchase"
]

cluster_data = data[cluster_cols].replace(
    [np.inf, -np.inf], np.nan
)

imputer = SimpleImputer(strategy="median")
cluster_array = imputer.fit_transform(cluster_data)

# Limit clusters when there are few customers
n_clusters = min(4, len(data))

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(cluster_array)
data.to_csv("customer_ml_results.csv", index=False)

# --------------------------------------------------
# 10. SAVE MODEL BUNDLE FOR STREAMLIT
# --------------------------------------------------
bundle = {
    "model": model,
    "features": feature_cols,
    "history_cutoff": str(cutoff.date()),
    "future_period_end": str(max_date.date()),
    "cluster_model": kmeans,
    "cluster_imputer": imputer,
    "cluster_features": cluster_cols
}

joblib.dump(bundle, "best_model.joblib", compress=3)

size_mb = os.path.getsize("best_model.joblib") / (1024 * 1024)

print("\nFILES SAVED")
print("-----------")
print("customer_features.csv")
print("customer_ml_results.csv")
print("model_predictions.csv")
print("feature_importance.csv")
print("model_comparison.csv")
print("best_model.joblib")
print(f"Model size: {size_mb:.2f} MB")
print(f"Total training time: {time.time() - start:.1f} seconds")

if size_mb > 25:
    print(
        "NOTE: Model is still over GitHub's 25 MB browser-upload "
        "limit. Keep the file and use another upload method."
    )
else:
    print("Model is under 25 MB and may be uploaded via the browser.")