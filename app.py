import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Load the model and transformers from the files
logistic_model = joblib.load('logistic_regression.joblib')
random_forest_model = joblib.load('random_forest_classifier.joblib')

label_encoder = joblib.load('label_encoder.joblib')
one_hot_encoder = joblib.load('one_hot_encoder.joblib')
standard_scaler = joblib.load('standard_scaler.joblib')

# Title
st.title('Heart Attack Risk Prediction')

# Function to get input and perform feature engineering
def get_input():
    st.sidebar.header('User Input')

    # Getting user input (normal)
    diabetes = int(st.sidebar.selectbox('Diabetes', [0, 1]))
    family_history = int(st.sidebar.selectbox('Family History', [0, 1]))
    smoking = int(st.sidebar.selectbox('Smoking', [0, 1]))
    obesity = int(st.sidebar.selectbox('Obesity', [0, 1]))
    alcohol_consumption = int(st.sidebar.selectbox('Alcohol Consumption', [0, 1]))
    exercise_hours_per_week = float(st.sidebar.slider('Exercise Hours Per Week', 0.0, 24.0, 0.0))
    previous_heart_problems = int(st.sidebar.selectbox('Previous Heart Problems', [0, 1]))
    medication_use = int(st.sidebar.selectbox('Medication Use', [0, 1]))
    stress_level = int(st.sidebar.slider('Stress Level', 0, 10, 0))
    sedentary_hours_per_day = float(st.sidebar.slider('Sedentary Hours Per Day', 0.0, 24.0, 0.0))
    physical_activity_days_per_week = int(st.sidebar.selectbox('Physical Activity Days Per Week', [0, 1, 2, 3, 4, 5, 6, 7]))
    sleep_hours_per_day = int(st.sidebar.slider('Sleep Hours Per Day', 0, 24, 0))

    # Label encoding - diet
    diet = st.sidebar.selectbox('Diet', ['Average', 'Unhealthy', 'Healthy'])
    diet_encoded = label_encoder.transform([diet])

    # One hot encoding - 'Sex', 'Continent', 'Hemisphere'
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    continent = st.sidebar.selectbox('Continent', ['Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America'])
    hemisphere = st.sidebar.selectbox('Hemisphere', ['Northern Hemisphere', 'Southern Hemisphere'])

    categorical_inputs = np.array([[sex, continent, hemisphere]])
    categorical_input_encoded = one_hot_encoder.transform(categorical_inputs).toarray()

    # Scaling numerical data
    age = float(st.sidebar.slider('Age', 0.0, 100.0, 0.0))
    cholesterol = float(st.sidebar.slider('Cholesterol', 0.0, 1000.0, 0.0))
    heart_rate = float(st.sidebar.slider('Heart Rate', 0.0, 200.0, 0.0))
    income = float(st.sidebar.slider('Income', 0.0, 1000000.0, 0.0))
    bmi = float(st.sidebar.slider('BMI', 0.0, 100.0, 0.0))
    blood_pressure_n = float(st.sidebar.slider('Blood Pressure N', 0.0, 200.0, 0.0))
    blood_pressure_d = float(st.sidebar.slider('Blood Pressure D', 0.0, 200.0, 0.0))

    numerical_inputs = np.array([[age, cholesterol, heart_rate, income, bmi, blood_pressure_n, blood_pressure_d]])
    numerical_inputs_scaled = standard_scaler.transform(numerical_inputs)

    # Combine all inputs
    final_features = np.hstack((
        [diabetes, family_history, smoking, obesity, alcohol_consumption, exercise_hours_per_week, 
         previous_heart_problems, medication_use, stress_level, sedentary_hours_per_day, 
         physical_activity_days_per_week, sleep_hours_per_day],
        diet_encoded,
        categorical_input_encoded.ravel(),
        numerical_inputs_scaled.ravel()
    ))

    # Reshape to match model input shape (1, -1)
    return final_features.reshape(1, -1)

# Get input
input_data = get_input()

# Predicting part
if st.button('Predict using Logistic Regression'):
    try:
        st.subheader('User Input Data')
        input_df = pd.DataFrame(input_data, columns=[
            'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
            'Exercise Hours Per Week', 'Previous Heart Problems', 'Medication Use', 
            'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 
            'Sleep Hours Per Day', 'Diet', 'Sex_Male', 'Sex_Female', 'Continent_Africa', 
            'Continent_Asia', 'Continent_Australia', 'Continent_Europe', 
            'Continent_North America', 'Continent_South America', 
            'Hemisphere_Northern Hemisphere', 'Hemisphere_Southern Hemisphere', 
            'Age', 'Cholesterol', 'Heart Rate', 'Income', 'BMI', 
            'Blood Pressure N', 'Blood Pressure D'
        ])
        st.write(input_df)


        prediction = logistic_model.predict(input_data)
        print(prediction)
        if prediction[0] == 1:
            st.error('You are at risk of heart attack')
        else:
            st.success('You are not at risk of heart attack')
    except Exception as e:
        st.error(f"Error: {e}")


if st.button('Predict using Random Forest Classifier'):
    try:
        st.subheader('User Input Data')
        input_df = pd.DataFrame(input_data, columns=[
            'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 
            'Exercise Hours Per Week', 'Previous Heart Problems', 'Medication Use', 
            'Stress Level', 'Sedentary Hours Per Day', 'Physical Activity Days Per Week', 
            'Sleep Hours Per Day', 'Diet', 'Sex_Male', 'Sex_Female', 'Continent_Africa', 
            'Continent_Asia', 'Continent_Australia', 'Continent_Europe', 
            'Continent_North America', 'Continent_South America', 
            'Hemisphere_Northern Hemisphere', 'Hemisphere_Southern Hemisphere', 
            'Age', 'Cholesterol', 'Heart Rate', 'Income', 'BMI', 
            'Blood Pressure N', 'Blood Pressure D'
        ])
        st.write(input_df)


        prediction = random_forest_model.predict(input_data)
        print(prediction)
        if prediction[0] == 1:
            st.error('You are at risk of heart attack')
        else:
            st.success('You are not at risk of heart attack')
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown('Developed by Saurav Karki')
st.markdown('Email: sauravkarki10.12@gmail.com')
st.markdown('LinkedIn: https://www.linkedin.com/in/saurav-karki-7b7a0b1b9/')
st.markdown('GitHub: http://github.com/itsmesauravk')
st.markdown('Medium: https://sauravkarki.medium.com')
