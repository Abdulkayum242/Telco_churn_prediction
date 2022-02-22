# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:19:58 2022

@author: onero
"""

import streamlit as st
import pickle


with open('grid_pkl','rb')  as f:
    model=pickle.load(f)





st.title('Telecom Churn Prediction')



partner=st.selectbox('Partner',(1,0))
if partner=='yes':
    partner=1
else:
     partner=0  



Dependents=st.selectbox('Dependents',(1,0))


IS=st.selectbox('Internet Service',(0,1,2))


osec=st.selectbox('Online Security',(1,0))


oback=st.selectbox('Online Backup',(1,0))


dprot=st.selectbox('Device Protection',(1,0))


tecsup=st.selectbox('Tech Support',(1,0))


stv=st.selectbox('Streaming TV',(1,0))


sm=st.selectbox('Streaming Movies',(1,0))

cont=st.selectbox('Contract',(0,1,2))


pbill=st.selectbox('Paperless Billing',(1,0))


paymet=st.selectbox('paymet',(0,1,2,3))


mc=st.number_input('Monthly Charges')

tc=st.number_input('Total Charges')



if st.button('Submit'):
    predict=model.predict([[partner,Dependents,IS,osec,oback,dprot,tecsup,stv,sm,cont,pbill,paymet,mc,tc]])
    if (predict[0])==0:
        st.write('Customer will not churn')
    else:
        st.write('Customer is likely to churn')
    












