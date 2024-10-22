import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc

# Load dataset
st.title("Diabetes Prediction Analysis")
st.write("This app analyzes a diabetes dataset using various machine learning models.")

uploaded_file = st.file_uploader("Upload your diabetes dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.write("Displaying the first few rows of the dataset:")
    st.dataframe(data.head())

    st.write("Displaying the last few rows of the dataset:")
    st.dataframe(data.tail())

    # Data preprocessing
    data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
    data["smoking_history"] = data["smoking_history"].map(
        {"never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6}
    )

    st.subheader("Data Description")
    st.write("Basic statistics of the dataset:")
    st.write(data.describe())

    st.subheader("Correlation Matrix")
    corr = data.corr()
    st.write(sns.heatmap(corr, annot=True))
    st.pyplot()

    # Box plot for data visualization
    st.subheader("Box Plot")
    st.write("Visualizing the spread of features:")
    st.write(data.plot(kind="box", figsize=(10, 5)))
    st.pyplot()

    # Feature and label separation
    y = data['diabetes']
    X = data.drop("diabetes", axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Model Training and Evaluation")

    # Logistic Regression
    if st.checkbox("Train Logistic Regression Model"):
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        st.write("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("F1 Score: ", f1_score(y_test, y_pred))
    
    # Random Forest Classifier
    if st.checkbox("Train Random Forest Model"):
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        st.write("Random Forest Accuracy: ", accuracy_score(y_test, y_pred_rf))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred_rf))
        st.write("F1 Score: ", f1_score(y_test, y_pred_rf))

        # Feature importance
        st.subheader("Feature Importance")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns

        plt.figure(figsize=(8, 6))
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), importances[indices], align="center", color="r")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        st.pyplot()
