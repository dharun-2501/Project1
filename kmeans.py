import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Clustering App", layout="centered")

# Load the trained KMeans model
with open("kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔍 Flexible K-Means Clustering App")
st.markdown("Upload any dataset and choose two numeric columns for clustering and visualization.")

# Upload file
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Let user select columns dynamically
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_columns) < 2:
        st.warning("Please upload a CSV with at least 2 numeric columns.")
    else:
        col1 = st.selectbox("Select Feature 1", numeric_columns)
        col2 = st.selectbox("Select Feature 2", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)

        if st.button("🔍 Cluster & Visualize"):
            x = df[[col1, col2]].values

            # Predict clusters
            labels = model.predict(x)
            df['Cluster'] = labels

            # Display clustered data
            st.subheader("📌 Clustered Data")
            st.dataframe(df)

            # Visualize clusters
            st.subheader("📈 Cluster Visualization")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[col1], y=df[col2], hue=df['Cluster'], palette='Set2', s=100)
            plt.title("Clustered Data")
            plt.xlabel(col1)
            plt.ylabel(col2)
            st.pyplot(plt)
