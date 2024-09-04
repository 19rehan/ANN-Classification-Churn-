import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder


#LOAD THE TRAINED MODEL

model = tf.keras.models.load_model('model.h5')


#LOAD THE ENCODERS ADN SCALERS


with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scaler = pickle.load(file)


#STREAMLIT APP
st.title('Customer Churn Prediction')    

#USER INPUT

geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18 , 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated_Salaray')
tenure = st.slider('Tenure',0 ,10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#Prepaer input Data

input_data = pd.DataFrame({
'CreditScore' : [credit_score],
'Gender' : [label_encoder_gender.transform([gender])[0]],
'Age' : [age],
'Tenure' : [tenure],
'Balance' : [balance],
'NumOfProducts': [num_of_products],
'HasCrCard' : [has_cr_card],
'IsActiveMember' : [is_active_member],
'EstimatedSalary' : [estimated_salary]
})


#ONEHOT ENCODE GEOGRAPHY

geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


#COMBINE ONEHOT ENCODED COLUMN WITH INPUT DATA


#input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis = 1)
input_df = pd.DataFrame([input_data])

input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])


input_df = pd.concat([input_df.drop('Geography',axis = 1),geo_encoded_df],axis=1)

#SCALING THE INPUT DATA

input_scaled = scaler.transform(input_df)

#PREDICT CHURN
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]


st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
 st.write("the customer is likely to churn.")
else:
 st.write('The customer is not likely to churn.')