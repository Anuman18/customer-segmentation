import streamlit as st
import pandas as pd
import joblib

# Title
st.title("🧠 Customer Segmentation App")
st.write("Upload your customer data and get cluster labels based on purchasing behavior.")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check necessary columns
    required_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
    if all(col in df.columns for col in required_cols):
        # Encode Gender
        df['Gender_Male'] = df['Gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)

        # Prepare input
        X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]

        # Load model & scaler
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Scale and predict
        X_scaled = scaler.transform(X)
        df['Cluster'] = kmeans.predict(X_scaled)

        # Show result
        st.success("✅ Segmentation Complete!")
        st.dataframe(df)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

    else:
        st.warning("Please ensure the file contains: Age, Annual Income (k$), Spending Score (1-100), and Gender.")
