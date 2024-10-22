import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

# Map categorical values
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

# Define features and labels
y = data['diabetes']
x = data.drop("diabetes", axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction")

# Create sliders and inputs for all features
age = st.slider("Age", 0, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smoking_history = st.selectbox("Smoking History", ["Never", "No Info", "Current", "Former", "Ever", "Not Current"])

# Add more feature inputs here as needed
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5)
blood_glucose = st.slider("Blood Glucose Level", 50, 250, 100)

# Convert gender and smoking history to numerical values
gender_map = {"Male": 1, "Female": 2, "Other": 3}
smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}

input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], 
                        hypertension, heart_disease, bmi, hba1c, blood_glucose]])

# Make predictions
if st.button("Predict"):
    prediction = rf_model.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: Have diabetes.")
    else:
        st.write("Prediction: Don't have  diabetes.")

# Feature Importance Plot
st.subheader("Feature Importance")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(8, 6))
plt.title("Feature Importance")
plt.bar(range(x.shape[1]), importances[indices], align="center", color="r")
plt.xticks(range(x.shape[1]), features[indices], rotation=45)
plt.tight_layout()

st.pyplot(plt)













