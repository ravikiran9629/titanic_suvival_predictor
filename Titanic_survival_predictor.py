import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the transformations and the trained model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ohe.pkl', 'rb') as f:
    OHE = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit app
st.title('Titanic Survival Predictor')

age = st.number_input('Age')
fare = st.number_input('Fare')
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Embarked', ['Cherbourg', 'Queenstown', 'Southampton'])
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sibsp = st.slider('Siblings/Spouse', 0, 10, 0)
parch = st.slider('Parents/Childern', 0, 10, 0)
    
data = {
    'Pclass': [pclass],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Female': [1 if sex == 'female' else 0],
    'Male': [1 if sex == 'Male' else 0],
    'C': [1 if embarked == 'Cherbourg' else 0],
    'Q': [1 if embarked == 'Queenstown' else 0],
    'S': [1 if embarked == 'SouthamptonS' else 0],
    'Age': [age],
    'Fare': [fare]
}
input_data = pd.DataFrame(data,index=[0])



# Standardize the input data
input_data[['Age','Fare']] = scaler.transform(input_data[['Age','Fare']] )

st.write("User-Provided Data")
st.dataframe(input_data)

# Predict
survival_proba = loaded_model.predict_proba(input_data)

st.write("User-Provided Data")
st.dataframe(input_data)


if st.button("Show Result"):
    # Code to execute after button click
    if survival_proba[0][1] > 0.5:
        st.write("Survived")
    else:
        st.write("Not Survived")
