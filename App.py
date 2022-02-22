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



partner=st.selectbox('partner',('yes','no'))
if partner=='yes':
    partner=1
else:
     partner=0  



Dependents=st.selectbox('Dependents',('yes','no'))
if Dependents=='yes':
    Dependents=1
else:
     Dependents=0  


Internet Service=st.selectbox('Internet Service',('DSL','Fiber optic','No'))
if Internet Service=='DSL':
    Internet Service=0
elif Internet Service=='Fiber optic':
    Internet Service=1
else:
    Internet Service=2



Online Security=st.selectbox('Online Security',('yes','no'))
if Online Security=='yes':
    Online Security=1
else:
     Online Security=0


Online Backup=st.selectbox('Online Backup',('yes','no'))
if Online Backup=='yes':
    Online Backup=1
else:
     Online Backup=0

Device Protection=st.selectbox('Device Protection',('yes','no'))
if Device Protection=='yes':
    Device Protection=1
else:
     Device Protection=0

Tech Support=st.selectbox('Tech Support',('yes','no'))
if Tech Support=='yes':
    Tech Support=1
else:
     Tech Support=0

Streaming TV=st.selectbox('Streaming TV',('yes','no'))
if Streaming TV=='yes':
    Streaming TV=1
else:
     Streaming TV=0

Streaming Movies=st.selectbox('Streaming Movies',('yes','no'))
if Streaming Movies=='yes':
    Streaming Movies=1
else:
     Streaming Movies=0

Contract=st.selectbox('Contract',('Month-to-month','One year','Two year'))
if Contract=='Month-to-month':
    Contract=0
elif Contract=='One year':
    Contract=1
else:
    Contract=2


Paperless Billing=st.selectbox('Paperless Billing',('yes','no'))
if Paperless Billing=='yes':
    Paperless Billing=1
else:
     Paperless Billing=0

payment Method=st.selectbox('payment Method',('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
if payment Method==='Electronic check':
     payment Method=0
 elif  payment Method=='Mailed check':
    payment Method=1
 elif payment Method=='Bank transfer (automatic)':
    payment Method=2            
else:
    payment Method=3

mc=st.number_input('Monthly Charges')

tc=st.number_input('Total Charges')



if st.button('Submit'):
    predict=model.predict([[partner,Dependents,IS,osec,oback,dprot,tecsup,stv,sm,cont,pbill,paymet,mc,tc]])
    if (predict[0])==0:
        st.write('Customer will not churn')
    else:
        st.write('Customer is likely to churn')
    












