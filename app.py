import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🛍️ Advanced Customer Segmentation")

# 2. Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["1. Dataset Overview", "2. Visual Analysis (EDA)", "3. K-Means Clustering", "4. Marketing Insights"])

uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # PAGE 1: Dataset Overview
    if page == "1. Dataset Overview":
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Avg. Income", f"${df['Annual Income (k$)'].mean():.2f}k")
        col3.metric("Avg. Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
        
        st.write("### Raw Data Preview")
        st.dataframe(df.head())
        st.write("### Statistical Description")
        st.dataframe(df.describe())

    # PAGE 2: Exploratory Analysis
    elif page == "2. Visual Analysis (EDA)":
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'], kde=True, color='skyblue', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.write("#### Income vs Spending")
            fig, ax = plt.subplots()
            sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, ax=ax)
            st.pyplot(fig)

    # PAGE 3: K-Means Clustering
    elif page == "3. K-Means Clustering":
        st.subheader("Machine Learning Logic")
        
        # 3a. Elbow Method
        st.write("#### 1. Finding Optimal Clusters (Elbow Method)")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        
        # 3b. Apply K-Means based on user input
        st.write("#### 2. Apply Segmentation")
        k = st.slider("Select K Clusters", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        
        # Plot clusters and centroids
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='Annual Income (k$)',
            y='Spending Score (1-100)',
            hue='Cluster',
            palette='Set1',
            data=df,
            ax=ax
        )
        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=300,
            c='black',
            marker='X',
            label='Centroids'
        )
        ax.legend()
        st.pyplot(fig)

    # PAGE 4: Marketing Insights
    elif page == "4. Marketing Insights":
        st.subheader("Target Groups Discovery")
        
        # High Spend/Low Income customers
        priority = df[(df['Spending Score (1-100)'] > 70) & (df['Annual Income (k$)'] < 40)]
        st.success(f"**High Potential Group:** We found {len(priority)} customers who spend heavily despite lower income.")
        st.dataframe(priority)
        
        # Download segmented data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Segmented Data", data=csv, file_name="segmented_customers.csv", mime="text/csv")

else:
    st.info("Please upload the Mall_Customers.csv file from the sidebar to begin.")
