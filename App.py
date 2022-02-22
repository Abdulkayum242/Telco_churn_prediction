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


Internet_Service=st.selectbox('Internet_Service',('DSL','Fiber optic','No'))
if Internet_Service=='DSL':
    Internet_Service=0
elif Internet_Service=='Fiber optic':
    Internet_Service=1
else:
    Internet_Service=2



Online_Security=st.selectbox('Online_Security',('yes','no'))
if Online_Security=='yes':
    Online_Security=1
else:
     Online_Security=0


Online_Backup=st.selectbox('Online_Backup',('yes','no'))
if Online_Backup=='yes':
    Online_Backup=1
else:
     Online_Backup=0

Device_Protection=st.selectbox('Device_Protection',('yes','no'))
if Device_Protection=='yes':
    Device_Protection=1
else:
     Device_Protection=0

Tech_Support=st.selectbox('Tech_Support',('yes','no'))
if Tech_Support=='yes':
    Tech_Support=1
else:
     Tech_Support=0

Streaming_TV=st.selectbox('Streaming_TV',('yes','no'))
if Streaming_TV=='yes':
    Streaming_TV=1
else:
     Streaming_TV=0

Streaming_Movies=st.selectbox('Streaming_Movies',('yes','no'))
if Streaming_Movies=='yes':
    Streaming_Movies=1
else:
     Streaming_Movies=0

Contract=st.selectbox('Contract',('Month-to-month','One year','Two year'))
if Contract=='Month-to-month':
    Contract=0
elif Contract=='One year':
    Contract=1
else:
    Contract=2


Paperless_Billing=st.selectbox('Paperless_Billing',('yes','no'))
if Paperless_Billing=='yes':
    Paperless_Billing=1
else:
     Paperless_Billing=0

payment_Method=st.selectbox('payment_Method',('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
if payment_Method=='Electronic check':
     payment_Method=0
elif  payment_Method=='Mailed check':
     payment_Method=1
elif payment_Method=='Bank transfer (automatic)':
     payment_Method=2            
else:
    payment_Method=3

mc=st.number_input('Monthly Charges')

tc=st.number_input('Total Charges')



if st.button('Submit'):
    predict=model.predict([[partner,Dependents,Internet_Service,Online_Security,Online_Backup,Device_Protection,Tech_Support,Streaming_TV,Streaming_Movies,Contract,Paperless_Billing,payment_Method,mc,tc]])
    if (predict[0])==0:
        st.write('Customer will not churn')
    else:
        st.write('Customer is likely to churn')
    












