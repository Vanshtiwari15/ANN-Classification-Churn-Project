import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time

## Load the trained model
model = tf.keras.models.load_model("model.h5")

## Load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title('📊 Customer Churn Prediction')
st.write("Fill in the customer details and click **Predict** to see if they will churn.")


# User input
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92)
balance = st.number_input('💰 Balance')
credit_score = st.number_input('🏦 Credit Score')
estimated_salary = st.number_input('💼 Estimated Salary')
tenure = st.slider('📅 Tenure', 0, 10)
num_of_products = st.slider(' 📦Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])



# Button to trigger prediction
if st.button("🚀 Predict Churn"):
    with st.spinner("🤖 Analyzing customer data..."):
        time.sleep(1.5)  # just for effect

        ## Prepare the input Data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        ## Prediction churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.write(f'Churn Probabilty: {prediction_proba:.2f}')
        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
