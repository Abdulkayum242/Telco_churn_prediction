# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:19:58 2022

@author: onero
"""

import streamlit as st

st.title('Telecom Churn Prediction')


st.selectbox('Dependents',('Yes','No'))

st.selectbox('Partner',('Yes','No'))


st.selectbox('Internet Service',('Fiber optic','DSL','No'))


st.selectbox('Online Security',('Yes','No'))


st.selectbox('Online Backup',('Yes','No'))


st.selectbox('Device Protection',('Yes','No'))


st.selectbox('StreamingTV',('Yes','No'))

st.selectbox('OnlineBackup',('Yes','No'))

st.selectbox('Streaming Movies',('Yes','No'))

st.selectbox('Contract',('Monthly','One year','Two year'))


st.selectbox('Paperless Billing',('Yes','No'))


st.selectbox('Payment Method',('Electronic check','Mailed check ','Bank transfer (automatic)','Credit card (automatic)'))


st.number_input('Monthly Charges')

st.number_input('Total Charges')



st.button('Submit')














