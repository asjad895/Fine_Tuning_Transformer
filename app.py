import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Function to load the model and tokenizer
@st.cache_data()
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("asjadiiit/finetunedBert_toxiccom_class")
    return tokenizer, model

# Load the tokenizer and model
tokenizer, model = get_model()

# Page title and description
st.title("Toxic Comment Classification")
st.markdown("This web application uses a fine-tuned BERT model to classify comments as toxic or non-toxic.")

# User input and analyze button
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

# Dictionary for class labels
class_labels = {
    1: 'Toxic',
    0: 'Non-Toxic'
}

if user_input and button:
    # Tokenize and prepare the input
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Make the prediction
    output = model(**test_sample)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    # Display the prediction and probabilities
    st.subheader("Prediction:")
    st.success(class_labels[predicted_class])
    st.subheader("Probability:")
    st.success(f"Toxic: {probabilities[0][1]:.4f}")
    st.success(f"Non-Toxic: {probabilities[0][0]:.4f}")
