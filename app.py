import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import streamlit.components.v1 as components

# Load the trained model
model = keras.models.load_model("career_nn_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load processed dataset to get available skills
df = pd.read_csv("processed_career_dataset.csv")
skill_columns = list(df.columns[1:])  # Exclude 'Career' column

# Set page configuration and custom styling
st.set_page_config(page_title="Career Predictor", page_icon="🌟", layout="wide")

# Brief references and model overview
st.write("**Dataset Reference:** Hossain Faruque, Sakir; Khushbu, Sharun Akter (2024), 'Student Career Dataset', Mendeley Data, V2, doi: 10.17632/4spj4mbpjr.2")
st.write("**Our Mission:** Helping you discover your best career path through personalized skill-based recommendations.")
st.write("**Model Training:** This neural network was trained on a labeled dataset and achieves approximately **94% accuracy** on test data.")

st.markdown(
    """
    <style>
        body {
            background-color: #b3e0ff;
            background-image: url('https://www.transparenttextures.com/patterns/bubbles.png');
            color: #003366;
        }
        .stButton > button {
            background-color: #005b96;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stMultiselect > div {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌊 Career Predictor")
st.write("### Select your skills, and we'll predict your best career match!")

# Skill selection with dropdown
selected_skills = st.multiselect("Choose your skills:", skill_columns)

# Convert selected skills into model input format
input_vector = np.zeros(len(skill_columns))  # Create zero vector
for skill in selected_skills:
    if skill in skill_columns:
        input_vector[skill_columns.index(skill)] = 1  # Set 1 for selected skills

# Predict career
if st.button("🔮 Predict Career"):
    input_vector = input_vector.reshape(1, -1)  # Reshape for model
    prediction = model.predict(input_vector)
    predicted_career = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    st.success(f"🌟 Your predicted career: **{predicted_career}**")
