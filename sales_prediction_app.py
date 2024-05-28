import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_path = 'models/rf_model.pkl'
with open(model_path, 'rb') as file:
    rf_reg = pickle.load(file)

# Function to preprocess input data
def preprocess_input_data(tv, radio, newspaper):
    data = {'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]}
    new_data = pd.DataFrame(data)
    new_data['log_TV'] = np.log(new_data['TV'] + 1)
    new_data['log_Radio'] = np.log(new_data['Radio'] + 1)
    new_data['log_Newspaper'] = np.log(new_data['Newspaper'] + 1)
    new_data['total_ad_expenditure'] = new_data['TV'] + new_data['Radio'] + new_data['Newspaper']
    new_data['log_total_ad_expenditure'] = np.log(new_data['total_ad_expenditure'] + 1)
    return new_data[['log_TV', 'log_Radio', 'log_Newspaper']]

# Function to add background image
def add_bg_from_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
bg_image_url = 'https://your-image-url.com/image.jpg'  # Replace with your image URL
add_bg_from_url(bg_image_url)

# Streamlit interface
st.title('Sales Prediction')
st.markdown("""
Welcome to the Sales Prediction! This tool leverages machine learning to predict the sales of a product based on advertising expenditures in various media channels. 

### Instructions:
1. **Enter the Advertising Budget**: Use the sidebar to input the budget allocated to TV, Radio, and Newspaper advertising.
2. **Predict Sales**: Once you have entered the budget, this will use the pre-trained model to predict the sales.
3. **View Results**: The predicted sales figure will be displayed on the main screen.

This tool helps businesses optimize their advertising strategies by predicting the potential sales based on different advertising budgets.
""")

st.sidebar.header('User Input Parameters')
def user_input_features():
    tv = st.sidebar.number_input('TV Advertising Budget', min_value=0.0, max_value=1000.0, value=0.0)
    radio = st.sidebar.number_input('Radio Advertising Budget', min_value=0.0, max_value=1000.0, value=0.0)
    newspaper = st.sidebar.number_input('Newspaper Advertising Budget', min_value=0.0, max_value=1000.0, value=0.0)
    data = {'TV': tv, 'Radio': radio, 'Newspaper': newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the input data
processed_data = preprocess_input_data(input_df['TV'][0], input_df['Radio'][0], input_df['Newspaper'][0])

# Make predictions
prediction = rf_reg.predict(processed_data)
st.subheader('Predicted Sales')
st.write(prediction[0])
