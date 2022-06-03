# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:07:24 2022

@author: diego
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import datetime

st.title('Copay and Billing Claim Status Prediction App.')
st.subheader('Erdos Bootcamp, May 2022. Group Drugs4All.')

@st.cache
def load_similar_drugs():
    similar_drugs = pd.read_csv('similar_drugs.csv')
    return similar_drugs

def list_similar_drugs(similar_drugs, diagnosis):
    diag_l, diag_n1, diag_n2 = split_diagnosis(diagnosis)
    list_1 = similar_drugs[(similar_drugs.diag_letter == diag_l) & (similar_drugs.diag_num1 == diag_n1) & (similar_drugs.diag_num2 == diag_n2)].drop(columns=['count']).reset_index(drop=True)
    list_2 = similar_drugs[(similar_drugs.diag_letter == diag_l) & (similar_drugs.diag_num1 == diag_n1)].drop(columns=['count']).reset_index(drop=True)
    list_3 = similar_drugs[(similar_drugs.diag_letter == diag_l)].drop(columns=['count']).reset_index(drop=True)
    return list_1, list_2, list_3

def split_diagnosis(diagnosis):
    diag_l = str(diagnosis.split('.')[0])
    diag_n1 = int(diagnosis.split('.')[1])
    diag_n2 = int(diagnosis.split('.')[2])
    return diag_l, diag_n1, diag_n2

def load_reg():
    model = load(r'./models/LGBMRegressor_-6.716346039146029.joblib')
    return model

def features():
    dep_features = 'patient_pay'
    ind_features = ['month_name', 
                    'day_name', 
                    'bin_pcn_group', 
                    'drug_brand', 
                    'drug_name', 
                    'diag_letter']
    return ind_features, dep_features

def predict_copay_list(model, df, month_name, day_name, bin_pcn_group, diag_l):
    for brand in ['branded', 'generic']:
        X = []
        for _,r in df.iterrows():
            X.append({'month_name':month_name,
                    'day_name':day_name,
                    'bin_pcn_group':bin_pcn_group, 
                    'drug_brand':brand,
                    'drug_name':r.drug_name, 
                    'diag_letter':diag_l})
        X = pd.DataFrame(X).astype('category')
        y_pred = model.predict(X)
        df['$_copay_'+brand] = np.around(y_pred,2)
    return df

data = load_similar_drugs()

st.subheader("Enter patient's information.")
diagnosis = st.text_input('Diagnosis', help='LETTER.NUMBER.NUMBER')
bin_id = st.text_input('BIN')
pcn = st.text_input('PCN')
group = st.text_input('Group')
drug_name = st.text_input('Drug Name')

enter_btn = st.button('Enter')

if enter_btn:
    if pcn and not group:
        group = 'NA'
    elif not pcn and group:
        pcn = 'NA'
    
    month_name = datetime.datetime.now().strftime('%B')
    day_name = datetime.datetime.now().strftime('%A')
    
    bin_pcn_group = '_'.join([bin_id, pcn, group])
    
    diag_l = diagnosis[0]
    
    # ind_features, dep_features = features()
    model = load_reg()
    y_pred_list = []
    for drug_brand in ['branded', 'generic']:
        X = pd.DataFrame([{'month_name':month_name,
                          'day_name':day_name,
                          'bin_pcn_group':bin_pcn_group, 
                          'drug_brand':drug_brand,
                          'drug_name':drug_name, 
                          'diag_letter':diag_l}]).astype('category')
        # X = pd.get_dummies(X)
        y_pred = model.predict(X)[0]
        y_pred_list.append(y_pred)
    st.text('Predicted copay for branded and generic {} are \n$ {:.2f} and $ {:.2f}, respectively.'.format(drug_name, y_pred_list[0], y_pred_list[1]))
    
    if diagnosis and bin_id and pcn and group and drug_name:
        list_1, list_2, list_3 = list_similar_drugs(data, diagnosis)
        similar_drugs = predict_copay_list(model, list_1, month_name, day_name, bin_pcn_group, diag_l)
        st.text('Similar drugs and expected copays:')
        st.write(similar_drugs)