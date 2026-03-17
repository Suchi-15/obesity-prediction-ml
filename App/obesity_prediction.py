import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
st.title('Obesity Prediction')
Age = st.text_input('Age:')
Gender = st.text_input('Gender:')
Height = st.text_input('Height:')
Weight = st.text_input('Weight:')
BMI = st.text_input('BMI:')
final_model=pickle.load(open('final_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
if Gender.lower() == 'male':
    Gender_encoded = 0
elif Gender.lower() == 'female':
    Gender_encoded = 1
#predict
if st.button('Predict'):
    a=scaler.transform([[int(Age), Gender_encoded, float(Height), float(Weight), float(BMI)]])
    yp= final_model.predict(a)
    print(yp)
    if yp[0]==0:
        st.write('Underweight')
    elif yp[0]==1:
        st.write('Normal')
    elif yp[0]==2:
        st.write('Overweight')
    elif yp[0]==3:
        st.write('Obese')