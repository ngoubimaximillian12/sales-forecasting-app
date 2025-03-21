# 📦 Imports
import kagglehub
import pandas as pd
import os
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine
import altair as alt

# 🎯 Streamlit Setup
st.set_page_config(page_title="📊 Sales Forecasting AI", layout="wide")
st.title("📊 Sales Forecasting AI System")
st.markdown("This app covers the full ML lifecycle: data → modeling → prediction → PostgreSQL → visualization → retraining")

# ✅ Step 1: Load Kaggle Dataset
try:
    path = kagglehub.dataset_download("rohitsahoo/sales-forecasting")
    csv_file = os.path.join(path, "train.csv")
    df = pd.read_csv(csv_file)
    st.success("✅ Dataset downloaded from Kaggle.")
except Exception as e:
    st.error(f"❌ Failed to download/load dataset: {e}")
    st.stop()

# ✅ Step 2: Clean + Prepare
try:
    target = "Sales"

    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.dropna(inplace=True)

    df_model = df.copy()
    df_viz = df.copy()

    # Encode object columns (excluding target)
    le = LabelEncoder()
    for col in df_model.select_dtypes(include=["object"]):
        if col != target:
            df_model[col] = le.fit_transform(df_model[col])

    if target not in df_model.columns:
        st.error(f"❌ Target column '{target}' not found. Columns: {df_model.columns.tolist()}")
        st.stop()

    st.success("🧹 Data cleaned and encoded.")

    # ✅ Step 3: Train or Load Model
    X = df_model.drop(target, axis=1)
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_path = "sales_forecast_model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("🤖 Model loaded from file.")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        st.success("✅ Model trained and saved.")

    # ✅ Step 4: Predict & Evaluate
    df_viz["Predicted_Sales"] = model.predict(df_model.drop(target, axis=1))
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.write(f"**MAE:** {mae:.2f} | **MSE:** {mse:.2f} | **R² Score:** {r2:.2f}")

except Exception as e:
    st.error(f"❌ Error during training/prediction: {e}")
    st.stop()

# ✅ Step 5: Visual Analytics
st.subheader("📊 Visual Analytics")

if "Category" in df_viz.columns:
    st.markdown("**🛒 Total Sales by Category**")
    bar_data = df_viz.groupby("Category")[target].sum().reset_index()
    st.altair_chart(alt.Chart(bar_data).mark_bar().encode(
        x=alt.X("Category:N", sort='-y'),
        y="Sales:Q",
        tooltip=["Category", "Sales"]
    ).properties(width=800))

if "Ship Date" in df_viz.columns:
    try:
        df_viz["Ship Date"] = pd.to_datetime(df_viz["Ship Date"])
        line_data = df_viz.groupby("Ship Date")[target].mean().reset_index()
        st.markdown("**📈 Avg Sales Over Time (by Ship Date)**")
        st.altair_chart(alt.Chart(line_data).mark_line(point=True).encode(
            x="Ship Date:T",
            y="Sales:Q"
        ).properties(width=800))
    except:
        st.warning("⚠️ 'Ship Date' column not in proper format.")

if "Region" in df_viz.columns:
    st.markdown("**🥧 Sales Distribution by Region**")
    pie_data = df_viz.groupby("Region")[target].sum().reset_index()
    st.altair_chart(alt.Chart(pie_data).mark_arc().encode(
        theta="Sales:Q",
        color="Region:N",
        tooltip=["Region", "Sales"]
    ).properties(width=500))

if "Sub-Category" in df_viz.columns:
    st.markdown("**📦 Sales Distribution by Sub-Category**")
    plt.figure(figsize=(12, 4))
    sns.boxplot(x="Sub-Category", y="Sales", data=df_viz)
    plt.xticks(rotation=90)
    st.pyplot(plt)

# Heatmap
st.markdown("**🔥 Correlation Heatmap**")
plt.figure(figsize=(10, 6))
sns.heatmap(df_model.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# Histogram
st.markdown("**📊 Histogram: Actual vs Predicted Sales**")
plt.figure(figsize=(10, 4))
sns.histplot(df_viz["Sales"], label="Actual", color="blue", kde=True)
sns.histplot(df_viz["Predicted_Sales"], label="Predicted", color="green", kde=True)
plt.legend()
st.pyplot(plt)

# ✅ Step 6: Upload to PostgreSQL
st.subheader("📤 Upload to PostgreSQL")

db_user = "postgres"
db_password = "hope"
db_host = "localhost"
db_port = "5432"
db_name = "SalesForecastingDB"

db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

if st.button("Upload Predictions"):
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute("SELECT 1")  # Connection test
        df_viz.to_sql("sales_forecast_predictions", engine, if_exists="replace", index=False)
        st.success("✅ Data uploaded to PostgreSQL table `sales_forecast_predictions` in database `SalesForecastingDB`.")
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")
        st.info("💡 Make sure PostgreSQL is running and your DB exists.")

# ✅ Step 7: Optional Retraining
st.subheader("🔁 Optional: Retrain Model on Current Data")

if st.button("Retrain Model"):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        st.success("🔁 Model retrained and saved.")
    except Exception as e:
        st.error(f"❌ Retraining failed: {e}")
