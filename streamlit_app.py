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

st.write('# THE ERDŐS INSTITUTE.\nData Science Bootcamp, May 2022. Group Drugs4All.')
st.write('June 4th, 2022.')
st.write('# Copay and Billing Claim Status Prediction App.')

def load_similar_drugs():
    drugs_df = pd.read_csv('similar_drugs.csv')
    drugs_df['diagnosis'] = drugs_df.diag_letter + drugs_df.diag_num1.astype('str') + '.' + drugs_df.diag_num2.astype('str')
    drugs_df = drugs_df.sort_values(by=['diag_letter','diag_num1','diag_num2', 'drug_name']).reset_index(drop=True)
    st.session_state.drugs_df = drugs_df
    
def load_unique_plans():
    unique_plans = pd.read_csv('unique_plans.csv')
    unique_plans = unique_plans.drop(columns=['count'])
    unique_plans = unique_plans.sort_values(by=['bin','pcn','group']).reset_index(drop=True)
    st.session_state.unique_plans = unique_plans

def load_reg():
    st.session_state.reg_model = load(r'./models/LGBMRegressor_-6.994210983740307.joblib')

def load_class():
    st.session_state.class_model = load(r'./models/LGBMClassifier_-7.630396256551146.joblib')

def callback_bin_box():
    unique_plans = st.session_state.unique_plans.copy()
    unique_plans = unique_plans.fillna('NA')
    options_pcn = unique_plans.pcn[unique_plans.bin == st.session_state.bin_box].sort_values().unique()
    options_group = unique_plans.group[unique_plans.bin == st.session_state.bin_box].sort_values().unique()
    st.session_state.options_pcn = options_pcn
    st.session_state.options_group = options_group
    return

def callback_submit_btn():
    if check_inputs():
        print_predict_copay_single()
        print_predict_copay_multi()

def create_form():
    st.write("### Start with patient's information.")
    drugs_df = st.session_state.drugs_df.copy()
    if 'options_group' not in st.session_state:
        st.session_state.options_group = ['']
    if 'options_pcn' not in st.session_state:
        st.session_state.options_pcn = ['']
        
    unique_plans = st.session_state.unique_plans.copy()
    
    diagnosis_box = st.selectbox('Diagnosis', ['Select Diagnosis']+list(drugs_df.diagnosis.unique()), key='diagnosis_box')
    bin_box = st.selectbox('BIN', ['Select BIN']+list(unique_plans.bin.unique()), key='bin_box', on_change=callback_bin_box, args=())
    pcn_box = st.selectbox('PCN', ['Select PCN']+list(st.session_state.options_pcn), key='pcn_box')
    group_box = st.selectbox('Group', ['Select Group']+list(st.session_state.options_group), key='group_box')
    drug_name_box = st.selectbox('Drug Name', ['Select Drug']+list(drugs_df.drug_name.sort_values().unique()), key='drug_name_box')
    submit_btn = st.button('Submit')

    join_bin_pcn_group()
    split_diagnosis()
    
    return submit_btn

def list_similar_drugs():
    similar_drugs = st.session_state.drugs_df.copy()
    st.session_state.list_1 = similar_drugs[(similar_drugs.diag_letter == st.session_state.diag_l) & (similar_drugs.diag_num1 == st.session_state.diag_n1) & (similar_drugs.diag_num2 == st.session_state.diag_n2)].drop(columns=['count']).reset_index(drop=True)
    st.session_state.list_2 = similar_drugs[(similar_drugs.diag_letter == st.session_state.diag_l) & (similar_drugs.diag_num1 == st.session_state.diag_n1)].drop(columns=['count']).reset_index(drop=True)
    st.session_state.list_3 = similar_drugs[(similar_drugs.diag_letter == st.session_state.diag_l)].drop(columns=['count']).reset_index(drop=True)

def join_bin_pcn_group():
    bin_id = str(st.session_state.bin_box)
    pcn = st.session_state.pcn_box if st.session_state.pcn_box != '' else 'NA'
    group = st.session_state.group_box if st.session_state.group_box != '' else 'NA'
    st.session_state.bin_pcn_group = '_'.join([bin_id, pcn, group])

def split_diagnosis():
    if 'Select' not in st.session_state.diagnosis_box:
        diagnosis = st.session_state.diagnosis_box
        st.session_state.diag_l = str(diagnosis[0])
        st.session_state.diag_n1 = int(diagnosis.split('.')[0][1:])
        st.session_state.diag_n2 = int(diagnosis.split('.')[1][0:])

def predict_copay_single():
    y_pred_list = []
    for brand in ['branded', 'generic']:
        X = []
        X.append({'bin_pcn_group':st.session_state.bin_pcn_group, 
                'drug_brand':brand,
                'drug_name':st.session_state.drug_name_box, 
                'diag_letter':st.session_state.diag_l})
        X = pd.DataFrame(X).astype('category')
        y_pred_class = st.session_state.class_model.predict(X)
        y_pred_reg = st.session_state.reg_model.predict(X)
        if y_pred_class[0]:
            y_pred_list.append(['Rejected', '0.00'])
        else:
            y_pred_list.append(['Approved', y_pred_reg[0]])
    return y_pred_list

def predict_copay_multi():
    list_similar_drugs()
    data = st.session_state.list_3.copy()
    for brand in ['branded', 'generic']:
        X = []
        for _,r in data.iterrows():
            X.append({
                      'bin_pcn_group':st.session_state.bin_pcn_group, 
                    'drug_brand':brand,
                    'drug_name':r.drug_name, 
                    'diag_letter':st.session_state.diag_l})
        X = pd.DataFrame(X).astype('category')
        y_pred_class = st.session_state.class_model.predict(X)
        y_pred_reg = st.session_state.reg_model.predict(X)
        data['status_'+brand] = np.where(np.array(y_pred_class) == False, 'Approved', 'Rejected')
        data['$_copay_'+brand] = y_pred_reg
        data['$_copay_'+brand].loc[data['status_'+brand] == 'Rejected'] = 0
        data['$_copay_'+brand] = data['$_copay_'+brand].round(2)
    return data

def print_predict_copay_single():
    y_pred_list = predict_copay_single()
    st.write('### Billing status and predicted copay for {}:'.format(st.session_state.drug_name_box))
    st.text('Branded: {} and $ {:.2f}.'.format(y_pred_list[0][0],
                                               y_pred_list[0][1]))
    st.text('Generic: {} and $ {:.2f}.'.format(y_pred_list[1][0],
                                               y_pred_list[1][1]))
    
def print_predict_copay_multi():
    similar_drugs = predict_copay_multi()
    similar_drugs = similar_drugs.sort_values(by=['diag_letter','diag_num1','diag_num2', 'drug_name']).reset_index(drop=True)
    similar_drugs = similar_drugs.drop(columns=['diag_letter','diag_num1','diag_num2'])
    similar_drugs = similar_drugs[['diagnosis','drug_name','status_branded','$_copay_branded','status_generic','$_copay_generic']]
    similar_drugs.columns = ['Diagnosis', 'Drug Name', 'Status if Branded', 'Copay $ if Branded', 'Status if Generic', 'Copay $ if Generic']
    st.write('### Expected billing status and copays for similar drugs:')
    st.write(similar_drugs)
    

def check_inputs():
    if 'Select' in str(st.session_state.diagnosis_box) or 'Select' in str(st.session_state.bin_box) or 'Select' in str(st.session_state.pcn_box) or 'Select' in str(st.session_state.group_box) or 'Select' in str(st.session_state.drug_name_box):
        return False
    
    # bin_id
    if not st.session_state.bin_box:
        st.warning('Enter a BIN ID.')
        return False
    
    # pcn and group
    if not st.session_state.pcn_box and not st.session_state.group_box:
        st.warning('Enter either PCN or Group.')
        return False
    
    # bin_id
    if not st.session_state.drug_name_box:
        st.warning('Enter a drug name.')
        return False
    else:
        if st.session_state.drug_name_box not in list(st.session_state.drugs_df['drug_name'].values):
            st.warning('The drug name is not available in our datasets. Try another one.')
            return False
    
    return True

if 'drugs_df' not in st.session_state:
    load_similar_drugs()
if 'unique_plans' not in st.session_state:
    load_unique_plans()
if 'reg_model' not in st.session_state:
    load_reg()
if 'class_model' not in st.session_state:
    load_class()
submit_btn = create_form()
if submit_btn:
    callback_submit_btn()