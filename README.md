# Shop Customer Clustering and Classification

## Overview
This project involves analyzing and classifying shop customer data to gain insights into customer behavior and improve business strategies. The project consists of two main components:

1. **Customer Clustering:** Grouping customers into clusters based on their characteristics.
2. **Customer Classification:** Building classification models to predict customer clusters based on specific features.

The customer data is sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/datascientistanna/customers-dataset).

---

## Customer Clustering Notebook

This notebook focuses on clustering customers based on their features. Below are the key steps:

### Steps Performed
1. **Introduction to Dataset:**
   - Overview of the dataset with details on rows, columns, and feature descriptions.

2. **Importing Libraries:**
   - Essential libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn` are imported.

3. **Loading the Dataset:**
   - Data is loaded into a Pandas DataFrame, and the first few rows are displayed.

4. **Exploratory Data Analysis (EDA):**
   - Dataset structure and descriptive statistics are examined.
   - Missing values are identified, and data visualizations are used to understand variable distributions.

5. **Data Preprocessing:**
   - Handling missing values by removing rows with null values.
   - Removing duplicate entries.
   - Detecting and handling outliers using the IQR method.
   - Encoding categorical features with `LabelEncoder`.

6. **Clustering Model Development:**
   - Implementing K-Means clustering with an initial number of clusters (K=3).
   - Evaluating the model using Silhouette Score.
   - Optimizing the number of clusters using the Elbow method and Silhouette Score.
   - Retraining the model with the optimal number of clusters.
   - Performing feature selection to identify influential features.
   - Training the K-Means model with selected features and comparing the results.

7. **Clustering Results Visualization:**
   - Visualizing clustering outcomes using PCA for dimensionality reduction.

8. **Cluster Analysis and Interpretation:**
   - Examining the characteristics of each cluster based on available features.
   - Displaying value distributions within each cluster.

9. **Exporting Results:**
   - Saving the clustering results to a CSV file.

This notebook provides a comprehensive analysis of shop customers, enabling the business to group customers into distinct segments for targeted strategies.

---

## Customer Classification Notebook

This notebook focuses on building machine learning models to classify customers into their respective clusters.

### Steps Performed
1. **Importing Libraries:**
   - Libraries such as `pandas`, `scikit-learn`, `seaborn`, and `matplotlib` are imported.

2. **Loading Clustered Dataset:**
   - The dataset from the clustering notebook is loaded into a DataFrame for further analysis.

3. **Data Splitting:**
   - The dataset is split into training (70%) and testing (30%) sets.

4. **Classification Model Development:**
   - Building models using the following algorithms:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors (K-NN)

5. **Model Evaluation:**
   - Evaluating models on the testing set using metrics such as:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix
   - Summary of results:
     - Decision Tree, Random Forest, and K-NN achieved perfect scores for all metrics (Accuracy, Precision, Recall, and F1-Score = 1.0).
     - Logistic Regression achieved an accuracy of 0.9119.

6. **Confusion Matrix Visualization:**
   - Confusion matrices for each model are visualized using `seaborn`.

This notebook demonstrates a step-by-step process for data analysis, model training, and performance evaluation for classifying shop customers.

---

## Conclusion
By combining clustering and classification approaches, this project provides valuable insights into customer segmentation and predictive analytics. These insights can be leveraged by businesses to create tailored marketing strategies, improve customer satisfaction, and optimize resource allocation.
