import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model_path = 'AutogluonModels/ag-20241206_093135'
predictor = TabularPredictor.load(model_path)
best_model_name = 'WeightedEnsemble_L2'

# Streamlit app
st.title('CitedByCount Prediction')

# Input form
st.header('Input Data')
year = st.number_input('Year', min_value=1900, max_value=2100, step=1)
title = st.text_input('Title')
publication_name = st.text_input('Publication Name')
author_keywords = st.text_input('Author Keywords')
asia = st.number_input('Asia', min_value=0, max_value=10, step=1)
oceania = st.number_input('Oceania', min_value=0, max_value=10, step=1)
europe = st.number_input('Europe', min_value=0, max_value=10, step=1)
north_america = st.number_input('North America', min_value=0, max_value=10, step=1)
africa = st.number_input('Africa', min_value=0, max_value=10, step=1)
south_america = st.number_input('South America', min_value=0, max_value=10, step=1)
unknown = st.number_input('Unknown', min_value=0, max_value=10, step=1)

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'Year': [year],
    'Title': [title],
    'PublicationName': [publication_name],
    'AuthorKeywords': [author_keywords],
    'Asia': [asia],
    'Oceania': [oceania],
    'Europe': [europe],
    'North America': [north_america],
    'Africa': [africa],
    'South America': [south_america],
    'Unknown': [unknown]
})

# Predict button
if st.button('Predict'):
    # Predict the CitedByCount
    prediction = predictor.predict(input_data, model=best_model_name)
    st.write(f'Predicted CitedByCount: {prediction[0]}')

    # Display the input data and prediction
    st.subheader('Input Data')
    st.write(input_data)

    st.subheader('Prediction')
    st.write(prediction)
