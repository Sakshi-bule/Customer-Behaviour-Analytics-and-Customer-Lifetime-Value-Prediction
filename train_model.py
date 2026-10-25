
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------------
# 1. Load and prepare transaction data
# -----------------------------------

print("Loading cleaned retail data...")

if not os.path.exists("cleaned_retail_data.csv"):
    raise FileNotFoundError(
        "cleaned_retail_data.csv not found. "
        "Run prepare_data.py first."
    )

df = pd.read_csv("cleaned_retail_data.csv")

required_columns = [
    "InvoiceNo", "InvoiceDate", "CustomerID",
    "Quantity", "UnitPrice", "Revenue"
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = df["CustomerID"].astype(str).str.replace(
    r"\.0$", "", regex=True
)
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

df = df.dropna(
    subset=["InvoiceNo", "InvoiceDate", "CustomerID", "Revenue"]
)
df = df[
    (df["Revenue"] > 0)
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
].copy()

if df.empty:
    raise ValueError("No valid transactions remain after cleaning.")

df["InvoiceDateOnly"] = df["InvoiceDate"].dt.normalize()

# -----------------------------------
# 2. Split transactions by date
# -----------------------------------

dates = np.sort(df["InvoiceDateOnly"].unique())

if len(dates) < 2:
    raise ValueError("At least two distinct transaction dates are required.")

split_index = int(len(dates) * 0.8)
split_index = min(max(split_index, 1), len(dates) - 1)
cutoff = pd.Timestamp(dates[split_index])

history = df[df["InvoiceDateOnly"] < cutoff].copy()
future = df[df["InvoiceDateOnly"] >= cutoff].copy()

if history.empty:
    raise ValueError("Historical transactions are empty.")

print("Historical cutoff:", cutoff.date())
print("Historical transactions:", len(history))
print("Future transactions:", len(future))

# -----------------------------------
# 3. Create customer features
# -----------------------------------

reference_date = history["InvoiceDate"].max() + pd.Timedelta(days=1)

features = history.groupby("CustomerID").agg(
    Total_Spending=("Revenue", "sum"),
    Purchase_Frequency=("InvoiceNo", "nunique"),
    Total_Quantity=("Quantity", "sum"),
    Average_Order_Value=("Revenue", "mean"),
    Last_Purchase=("InvoiceDate", "max")
)

features["Recency_Days"] = (
    reference_date - features["Last_Purchase"]
).dt.days

features = features.drop(columns=["Last_Purchase"])

# Future spending is the prediction target.
future_spending = future.groupby("CustomerID")["Revenue"].sum()
future_spending.name = "Future_Spending"

dataset = features.join(future_spending, how="left")
dataset["Future_Spending"] = dataset["Future_Spending"].fillna(0)

FEATURES = [
    "Total_Spending",
    "Purchase_Frequency",
    "Total_Quantity",
    "Average_Order_Value",
    "Recency_Days"
]

X = dataset[FEATURES].replace([np.inf, -np.inf], np.nan)
y = dataset["Future_Spending"].clip(lower=0)

# -----------------------------------
# 4. Customer segmentation
# -----------------------------------

print("Creating customer segments...")

segment_data = np.log1p(
    features[["Total_Spending", "Purchase_Frequency", "Total_Quantity"]]
    .clip(lower=0)
)

scaler = StandardScaler()
scaled_segments = scaler.fit_transform(segment_data)

n_clusters = min(3, len(features))

if n_clusters >= 2:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(scaled_segments)

    features["Cluster_ID"] = cluster_ids

    cluster_order = (
        features.groupby("Cluster_ID")["Total_Spending"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    if len(cluster_order) == 3:
        labels = {
            cluster_order[0]: "Low Value",
            cluster_order[1]: "Regular",
            cluster_order[2]: "High Value"
        }
    else:
        labels = {
            cluster_id: f"Segment {i + 1}"
            for i, cluster_id in enumerate(cluster_order)
        }

    features["Customer_Segment"] = features["Cluster_ID"].map(labels)
else:
    features["Cluster_ID"] = 0
    features["Customer_Segment"] = "All Customers"

# -----------------------------------
# 5. Build the Random Forest model
# -----------------------------------

print("Training Random Forest model...")

model_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# The target is log-transformed during training and automatically
# converted back to spending units during prediction.
model = TransformedTargetRegressor(
    regressor=model_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# Holdout evaluation, when enough customers are available.
if len(dataset) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, None)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

    print("\nModel evaluation")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
else:
    print("Few customers available; skipping holdout evaluation.")

# -----------------------------------
# 6. Train final model on all customers
# -----------------------------------

print("Training final model on all customers...")
model.fit(X, y)

# -----------------------------------
# 7. Save predictions and results
# -----------------------------------

dataset["Predicted_Future_Spending"] = np.clip(
    model.predict(X), 0, None
)

results = features.join(
    dataset[["Future_Spending", "Predicted_Future_Spending"]]
)

results = results.reset_index()
results.to_csv("customer_ml_results.csv", index=False)

prediction_output = results[
    ["CustomerID", "Future_Spending", "Predicted_Future_Spending"]
]
prediction_output.to_csv("model_predictions.csv", index=False)

# Save feature importance from the fitted Random Forest.
rf = model.regressor_.named_steps["model"]

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv("feature_importance.csv", index=False)

# -----------------------------------
# 8. Save the trained model
# -----------------------------------

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "history_cutoff": cutoff.strftime("%Y-%m-%d")
    },
    "best_model.joblib"
)

# Verify the model file exists.
if not os.path.exists("best_model.joblib"):
    raise RuntimeError("Model file was not created.")

print("\nTraining completed successfully!")
print("Saved customer_ml_results.csv")
print("Saved model_predictions.csv")
print("Saved feature_importance.csv")
print("Saved best_model.joblib")
print("Model file size:", round(
    os.path.getsize("best_model.joblib") / (1024 * 1024), 2
), "MB")
