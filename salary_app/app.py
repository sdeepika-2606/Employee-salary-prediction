import streamlit as st
import joblib

# Load model
model = joblib.load(open('salary_predictor_model.pkl', 'rb'))

# Streamlit App
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Predictor")
st.markdown("Predict your salary based on years of experience.")

# Input
experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)

# Predict
if st.button("Predict Salary 💰"):
    prediction = model.predict([[experience]])
    st.success(f"Predicted Salary: ₹{prediction[0]:,.2f}")
