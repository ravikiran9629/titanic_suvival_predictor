import streamlit as st
import pandas as pd
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
def user_input_features():
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
    'Cherbourg': [1 if embarked == 'C' else 0],
    'Queenstown': [1 if embarked == 'Q' else 0],
    'Southampton': [1 if embarked == 'S' else 0],
    'Age': [age],
    'Fare': [fare]
    }
    input_data = pd.DataFrame(data,index=[0])
    return input_data


# Standardize the input data
input_data[['Age','Fare']] = scaler.transform(input_data[['Age','Fare']] )

# Predict
survival_proba = loaded_model.predict_proba(input_data)
st.write('Yes' if survival_proba[0][1] > 0.5 else 'No')
