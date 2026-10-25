
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Customer Behaviour & CLV",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 1000

    df = pd.DataFrame({
        "Customer_ID": range(1, n + 1),
        "Age": rng.integers(18, 70, n),
        "Annual_Income": rng.integers(20000, 150000, n),
        "Purchase_Frequency": rng.integers(1, 30, n),
        "Average_Order_Value": rng.uniform(200, 5000, n),
        "Days_Since_Last_Purchase": rng.integers(1, 365, n),
        "Website_Visits": rng.integers(1, 100, n),
        "Discount_Usage": rng.integers(0, 20, n)
    })

    # Synthetic target for demonstrating the ML workflow
    engagement = 0.5 + 0.5 * np.exp(
        -df["Days_Since_Last_Purchase"] / 180
    )
    df["Lifetime_Value"] = (
        df["Purchase_Frequency"]
        * df["Average_Order_Value"]
        * engagement
        * rng.uniform(0.5, 1.5, n)
    )

    # Data mining: K-Means customer segmentation
    segment_cols = [
        "Purchase_Frequency",
        "Average_Order_Value",
        "Days_Since_Last_Purchase",
        "Website_Visits"
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[segment_cols])
    df["Customer_Segment"] = KMeans(
        n_clusters=3, random_state=42, n_init=10
    ).fit_predict(scaled)

    features = [
        "Age", "Annual_Income", "Purchase_Frequency",
        "Average_Order_Value", "Days_Since_Last_Purchase",
        "Website_Visits", "Discount_Usage"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["Lifetime_Value"],
        test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    }

    return df, model, features, metrics, y_test, pred


df, model, features, metrics, y_test, predictions = build_model()

st.title("📊 Customer Behaviour Analysis & Lifetime Value Prediction")
st.caption("CL-1 Project | Data Mining • Data Visualization • Machine Learning")
st.warning(
    "Demonstration project using synthetic data. "
    "The lifetime-value target is simulated and is not a real-world forecast."
)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Customer Analysis", "CLV Prediction", "ML Evaluation"]
)

if page == "Dashboard":
    st.header("Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Average Order Value", f"₹{df['Average_Order_Value'].mean():,.0f}")
    c3.metric("Average Simulated CLV", f"₹{df['Lifetime_Value'].mean():,.0f}")
    c4.metric("Customer Segments", str(df["Customer_Segment"].nunique()))

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        sns.histplot(df["Lifetime_Value"], kde=True, ax=ax)
        ax.set_title("Customer Lifetime Value Distribution")
        ax.set_xlabel("Simulated value (₹)")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()
        sns.countplot(
            data=df, x="Customer_Segment",
            hue="Customer_Segment", legend=False, ax=ax
        )
        ax.set_title("Customer Segment Distribution")
        ax.set_xlabel("K-Means cluster")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Customer Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button(
        "Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="customer_behaviour.csv",
        mime="text/csv"
    )

elif page == "Customer Analysis":
    st.header("Data Mining & Visualization")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df, x="Purchase_Frequency",
            y="Average_Order_Value", hue="Customer_Segment",
            palette="Set2", ax=ax
        )
        ax.set_title("K-Means Customer Segmentation")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df, x="Website_Visits",
            y="Lifetime_Value", ax=ax
        )
        ax.set_title("Website Visits vs Simulated CLV")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Customer Segment Profile")
    summary = df.groupby("Customer_Segment").agg(
        Customers=("Customer_ID", "count"),
        Avg_Frequency=("Purchase_Frequency", "mean"),
        Avg_Order_Value=("Average_Order_Value", "mean"),
        Avg_CLV=("Lifetime_Value", "mean"),
        Avg_Recency=("Days_Since_Last_Purchase", "mean")
    ).round(2)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df[features + ["Lifetime_Value"]].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cluster numbers are arbitrary. Interpret each segment using its profile."
    )

elif page == "CLV Prediction":
    st.header("Predict Customer Lifetime Value")
    st.write("Enter customer details to generate a model prediction.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input(
                "Annual Income (₹)", 0, 10000000, 60000, step=5000
            )
            frequency = st.number_input(
                "Purchase Frequency", 1, 1000, 15
            )
            order_value = st.number_input(
                "Average Order Value (₹)", 1.0, 1000000.0, 1500.0
            )

        with c2:
            recency = st.number_input(
                "Days Since Last Purchase", 0, 3650, 20
            )
            visits = st.number_input("Website Visits", 0, 10000, 40)
            discounts = st.number_input("Discount Uses", 0, 1000, 5)

        submitted = st.form_submit_button("Predict CLV")

    if submitted:
        customer = pd.DataFrame([{
            "Age": age,
            "Annual_Income": income,
            "Purchase_Frequency": frequency,
            "Average_Order_Value": order_value,
            "Days_Since_Last_Purchase": recency,
            "Website_Visits": visits,
            "Discount_Usage": discounts
        }])[features]

        predicted_value = max(0, model.predict(customer)[0])

        st.success(
            f"Estimated simulated 12-month value: ₹{predicted_value:,.2f}"
        )
        st.write("Prediction is an educational estimate from synthetic training data.")

elif page == "ML Evaluation":
    st.header("Machine Learning Model Evaluation")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"₹{metrics['MAE']:,.2f}")
    c2.metric("RMSE", f"₹{metrics['RMSE']:,.2f}")
    c3.metric("R² Score", f"{metrics['R2']:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.7)
    ax.set_xlabel("Actual simulated CLV (₹)")
    ax.set_ylabel("Predicted CLV (₹)")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)
    plt.close(fig)

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values()

    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **MAE:** Average absolute prediction error.  
    **RMSE:** Error metric that penalizes larger errors more strongly.  
    **R²:** Measures model fit relative to predicting the target mean.
    """)
