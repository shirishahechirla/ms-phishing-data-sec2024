import streamlit as st
import pickle
import re
import string
import numpy as np

# Step 1: Load the trained model and vectorizer from .pkl files
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Step 2: Function to clean the input text
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

# Step 3: Streamlit UI
st.set_page_config(page_title="Message Phishing Detector", page_icon="📧")

# Header section
st.title("📧 Message Phishing Detector")
st.markdown("""
This application helps you identify whether a message is **Phishing** (potentially harmful) or **Legitimate**.  
Simply enter your message below and click the **Predict** button to get the result.
""")

# User input
st.subheader("Enter Your Message")
user_input = st.text_area("Type your message here:", height=150)

# Add a Predict button
if st.button("Predict"):
    if user_input.strip():
        # Clean and preprocess the input
        cleaned_input = clean_text(user_input)  
        vectorized_input = vectorizer.transform([cleaned_input])  

        # Model prediction
        prediction = model.predict(vectorized_input)
        
        # Display the result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ **This message appears to be Phishing.** Avoid clicking on links or sharing sensitive information.")
        else:
            st.success("✅ **This message appears to be Legitimate.** No harmful indicators were detected.")
    else:
        st.warning("Please enter a message before clicking Predict.")