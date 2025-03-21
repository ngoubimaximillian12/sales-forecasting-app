# 📊 Sales Forecasting AI System (Streamlit + ML + PostgreSQL + Tableau)

This project is a complete end-to-end AI-powered sales forecasting system built with **Python**, **Streamlit**, **scikit-learn**, **PostgreSQL**, and **Tableau** for visualization.  
It automates data download, preprocessing, machine learning model training, evaluation, prediction, dashboarding, database export, and data sharing.

---

## 🚀 Features Overview

### ✅ Data Collection
- Downloads sales forecasting dataset from **Kaggle** using `kagglehub`.

### ✅ Data Cleaning & Preparation
- Removes duplicates and missing values
- Encodes categorical variables (label encoding)
- Prepares a DataFrame for ML and visualizations separately

### ✅ Model Training
- Uses **Random Forest Regressor** to predict `Item_Outlet_Sales`
- Handles train/test split and saves the model with `joblib`

### ✅ Model Evaluation
- Shows metrics: MAE, MSE, R²
- Evaluates accuracy on test data

### ✅ Visualizations (Streamlit + Altair + Seaborn + Matplotlib)
- 📊 Bar chart: Total Sales by Item Type
- 📈 Line chart: Sales over years
- 🥧 Pie chart: Sales distribution by outlet type
- 📦 Boxplot: Item type sales spread
- 🔥 Heatmap: Feature correlations
- 📉 Histogram: Actual vs Predicted Sales

### ✅ PostgreSQL Integration
- Uploads predictions to a **PostgreSQL** database
- Configured with credentials:

### ✅ Data Export
- 📥 **Download Predictions as CSV**
- 📥 **Download Predictions as Excel (.xlsx)**
- 🖨️ **Export full dashboard as PDF** (via browser Print → Save as PDF)

### ✅ Tableau Integration
- After predictions are uploaded to PostgreSQL, you can:
1. Open **Tableau Desktop**
2. Click **Connect** → **PostgreSQL**
3. Use:
   - Server: `localhost`
   - Port: `5432`
   - Database: `SalesForecastingDB`
   - Table: `sales_forecast_predictions`
   - Username: `postgres`
   - Password: `hope`
4. Load the table and build dashboards

> ⚠️ If you see a Tableau error like `SCRAM authentication not supported`, change PostgreSQL auth method from `scram-sha-256` to `md5` in `pg_hba.conf` and restart PostgreSQL.

### ✅ Optional: Model Retraining
- One-click retraining of the ML model using the current data

---

## 📁 Folder Structure

