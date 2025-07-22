# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")  # or LogisticRegression_model.pkl if not renamed

st.title("🌸 Iris Flower Classifier")
st.markdown("Predict the species of Iris flower based on measurements.")

# Get user inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare input
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=['sepal length (cm)', 'sepal width (cm)',
                                   'petal length (cm)', 'petal width (cm)'])

# Predict
if st.button("Predict Species"):
    prediction = model.predict(input_data)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"🌼 Predicted Species: **{species_map[prediction]}**")
