import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
descrip = pd.read_csv("symptom_Description.csv")
scaler = joblib.load("scaler.pkl")
X =  np.load('X.npy')
model = joblib.load("model.h5")
severity = pd.read_csv("severity.csv")
precautions = pd.read_csv("symptom_precaution.csv")
# st.write(list(precautions.loc[precautions['Disease'] == 'Malaria'].values))
symptoms = severity["Symptom"]
symps = []
for i in symptoms:
    symps.append(i)

st.write("## Disease Predictor Model")
inputs = st.multiselect('Select three known variables:', symps)
btn = st.button("Predict")
if btn :
   if len(inputs) <= 17:
       array = np.zeros((1, 17))
       weights = severity.loc[severity['Symptom'].isin(list(inputs))]['weight']
       weights = list(weights)
       for i in range(len(weights)):
           array[0,i]= weights[i]
       arr = scaler.transform(array)
       ans = model.predict(arr)
       conf = model.predict_proba(arr) 
       a = ans[0].replace("_"," ")
       st.write("# ",a)
       st.write("### ", "Description", ":-")
       desc = (descrip.loc[descrip['Disease'] == a].values)
       st.write(desc)
       precs = (precautions.loc[precautions['Disease'] == a].values)
       st.write("### ", "Precautions", ":-")
       st.write(precs)
       



   else:
       st.warning("You can select maximum 17 symptoms")
