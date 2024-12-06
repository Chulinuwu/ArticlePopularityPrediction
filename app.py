import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model_path = 'AutogluonModels/ag-20241206_093135'
predictor = TabularPredictor.load(model_path, require_py_version_match=False)
best_model_name = 'XGBoost'

# Streamlit app
st.title('CitedByCount Prediction')

# Input form
st.header('Input Data ( Model : Extreme Gradient Boosting)')
year = st.number_input('Year (ปีที่ตีพิมพ์)', min_value=1900, max_value=2100, step=1)
title = st.text_input('Title (หัวข้อบทความ)')
publication_name = st.text_input('Publication Name (ชื่อวารสาร)')
author_keywords = st.text_input('Author Keywords (คีย์เวิร์ดของผู้เขียน)')
authors_from_asia = st.number_input('authors_from_Asia (จำนวนผู้เขียนจากทวีป Asia)', min_value=0, max_value=10, step=1)
authors_from_oceania = st.number_input('authors_from_Oceania (จำนวนผู้เขียนจากทวีป Oceania)', min_value=0, max_value=10, step=1)
authors_from_europe = st.number_input('authors_from_Europe (จำนวนผู้เขียนจากทวีป Europe)', min_value=0, max_value=10, step=1)
authors_from_north_america = st.number_input('authors_from_North America (จำนวนผู้เขียนจากทวีป North America)', min_value=0, max_value=10, step=1)
authors_from_africa = st.number_input('authors_from_Africa (จำนวนผู้เขียนจากทวีป Africa)', min_value=0, max_value=10, step=1)
authors_from_south_america = st.number_input('authors_from_South America (จำนวนผู้เขียนจากทวีป America)', min_value=0, max_value=10, step=1)
authors_from_unknown = st.number_input('authors_from_Unknown (จำนวนผู้เขียนจากทวีปอะไรก็ไม่รู้)', min_value=0, max_value=10, step=1)

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'Year': [year],
    'Title': [title],
    'PublicationName': [publication_name],
    'AuthorKeywords': [author_keywords],
    'Asia': [authors_from_asia],
    'Oceania': [authors_from_oceania],
    'Europe': [authors_from_europe],
    'North America': [authors_from_north_america],
    'Africa': [authors_from_africa],
    'South America': [authors_from_south_america],
    'Unknown': [authors_from_unknown]
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
