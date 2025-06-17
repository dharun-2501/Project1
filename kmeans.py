import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Customer Segmentation using KMeans", layout="centered")

# Load the trained KMeans model
with open("kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🛍️  Customer Segmentation App")
st.markdown("Upload your dataset and view K-Means clusters based on **Annual Income** and **Spending Score**.")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    if 'Annual Income (k$)' in data.columns and 'Spending Score (1-100)' in data.columns:
        # Extract features for clustering
        x = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

        # Predict clusters using loaded model
        labels = model.predict(x)
        data['Cluster'] = labels

        # Show updated data
        st.subheader("📌 Clustered Data")
        st.dataframe(data)

        # Plot clusters
        st.subheader("📈 Cluster Visualization")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='Set2', s=100)
        plt.title("Customer Segments")
        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        st.pyplot(plt)
    else:
        st.error("The uploaded CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)' columns.")
