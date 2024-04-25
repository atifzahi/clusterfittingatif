#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Function to load the dataset
def load_data(file_path):
    """
    Load the dataset from the given file path.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    # Load the dataset, specifying NaN for missing values
    data = pd.read_csv(file_path, na_values=['NaN'])
    return data

# Function for data preprocessing
def preprocess_data(data):
    """
    Preprocess the dataset by dropping unnecessary columns and handling missing values.
    
    Parameters:
    data (pd.DataFrame): Input dataset.
    
    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    # Drop unnecessary columns
    data.drop(['Country', 'Year'], axis=1, inplace=True)
    
    # Handle missing values (if any)
    data.dropna(subset=['Club'], inplace=True)  # Remove rows with missing 'Club'
    
    return data

# Function for exploratory data analysis
def perform_eda(data):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    
    Parameters:
    data (pd.DataFrame): Input dataset.
    """
    # Summary statistics
    summary_stats = data.describe()
    print("Summary Statistics:\n", summary_stats)
    
    # Visualization: Relational Graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Shots', y='Goals')
    plt.title('Shots vs. Goals')
    plt.xlabel('Shots')
    plt.ylabel('Goals')
    plt.show()
    
     # Visualization: Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='Goals', bins=20, kde=True)
    plt.title('Distribution of Goals')
    plt.xlabel('Goals')
    plt.ylabel('Frequency')
    plt.show()
    
    # Visualization: Statistical Graph
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Function for clustering analysis
def perform_clustering(data):
    """
    Perform clustering analysis on the dataset.
    
    Parameters:
    data (pd.DataFrame): Input dataset.
    """
    # Select features for clustering
    features = ['Shots', 'Goals']
    X = data[features]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Fit KMeans clustering with optimal number of clusters
    k = 3  # Example: Change as needed
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to the dataset
    data['Cluster'] = kmeans.labels_
    
    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Shots', y='Goals', hue='Cluster', palette='viridis')
    plt.title('Clusters of Players')
    plt.xlabel('Shots')
    plt.ylabel('Goals')
    plt.show()

# Function for fitting analysis
def perform_fitting(data):
    """
    Perform fitting analysis on the dataset.
    
    Parameters:
    data (pd.DataFrame): Input dataset.
    """
    # Fit Linear Regression model
    X = data[['Shots']]
    y = data['Goals']
    model = LinearRegression()
    model.fit(X, y)
    
    # Visualize Fitted Model
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.title('Linear Regression: Shots vs. Goals')
    plt.xlabel('Shots')
    plt.ylabel('Goals')
    plt.show()
    
    return model

# Function for clustering quality evaluation
def clustering_quality(data):
    """
    Evaluate the quality of clustering.
    
    Parameters:
    data (pd.DataFrame): Input dataset with cluster labels.
    """
    # Silhouette Score
    silhouette_avg = silhouette_score(data[['Shots', 'Goals']], data['Cluster'])
    print("Silhouette Score:", silhouette_avg)

# Function for fitting quality evaluation
def fitting_quality(data, model):
    """
    Evaluate the quality of fitting.
    
    Parameters:
    data (pd.DataFrame): Input dataset for fitting analysis.
    model (object): Fitted Linear Regression model.
    """
    # Calculate Residual Sum of Squares (RSS)
    predicted_goals = model.predict(data[['Shots']])
    rss = np.sum((predicted_goals - data['Goals']) ** 2)
    
    # Calculate Total Sum of Squares (TSS)
    mean_goals = np.mean(data['Goals'])
    tss = np.sum((data['Goals'] - mean_goals) ** 2)
    
    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    print("R-squared:", r_squared)

# Main function
def main():
    # Load the dataset
    data = load_data("Data.csv")  # Replace "your_file_path.csv" with the actual path to your CSV file
    
    # Data preprocessing
    data = preprocess_data(data)
    
    # Exploratory Data Analysis (EDA)
    perform_eda(data)
    
    # Clustering Analysis
    perform_clustering(data)
    
    # Fitting Analysis
    model = perform_fitting(data)
    
    # Evaluate clustering quality
    clustering_quality(data)
    
    # Evaluate fitting quality
    fitting_quality(data, model)

# Execute the main function
if __name__ == "__main__":
    main()


# In[ ]:




