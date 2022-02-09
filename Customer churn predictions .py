#!/usr/bin/env python
# coding: utf-8

# In[96]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[206]:


cd "C:/Users/onero/Pictures/images/Model/Telco"


# In[97]:


# importing data sets 

df=pd.read_csv('Telco-Customer-Churn.csv')
pd.set_option("display.max_columns",None)


# In[98]:


df.head()


# In[99]:


df.shape


# In[100]:


df.info(verbose=True)


# In[101]:


#  percentage of null values

df.isnull().sum()/len(df)*100


# In[102]:


df.describe()


# Average monthly charged are USD 65 whereas 25% customers pay more than USD 90 per month.

# In[103]:


# Check the imbalance data set

df['Churn'].value_counts().plot(kind='barh',figsize=(8,8))
plt.xlabel('counts')
plt.ylabel('Target label')
plt.title('Countplot of yes/No ')


# In[104]:


df['Churn'].value_counts()/len(df)*100


# **The ratio is 73:27 which indicates the data is imbalanced** 

# In[105]:


mis_data=pd.DataFrame((df.isnull().sum()*100/df.shape[0]).reset_index())
mis_data


# In[106]:


plt.figure(figsize=(20,5))
sns.pointplot('index',0,data=mis_data)
plt.xticks(rotation=90,fontsize=15)
plt.title("percentage of missing value")


# * No missing value in data
# * Features with high missing values gives very less insight in analysis.
# 

# ### Data cleaning
# 
# 1 ) create the copy of data for manipulations and processing.
# 

# In[107]:


telco_df=df.copy()


# 2) Total charges should be numeric not object so let's convert it into numeric.

# In[108]:


telco_df.TotalCharges=pd.to_numeric(telco_df.TotalCharges,errors='coerce')


# In[109]:


telco_df.describe()


# In[110]:


telco_df.isnull().sum()/len(telco_df)*100


# As we can see 11 missing values are there in TotalCharges let's check this value

# In[111]:


telco_df.loc[telco_df['TotalCharges'].isnull()==True]


# 3) Missing value treatment

# * Only 15% of data is missing from Total charges compared to all data so it is safe to drop them for further processing.

# In[112]:


telco_df.dropna(how='any',inplace=True)


# In[113]:


telco_df['TotalCharges'].isnull().sum()


# 4) Divide customers into bins of tenure. for tenure<10 months assign a tenure group of 1-10 ,for tenure between 1-2years assign a tenure group of 13-24 and so on.

# In[114]:


# max tenure

telco_df['tenure'].max()


# In[115]:


# min tenure
telco_df['tenure'].min()


# so minimum no of months customers has subscribed is 1 and maximum is 72 months. 

# In[116]:


# Group the tenure into 12 months
labels=["{}-{}".format(i,i+11) for i in range(1,72,12)]
labels


# In[117]:


telco_df['tenure_group']=pd.cut(telco_df.tenure,range(1,80,12),right=False,labels=labels)


# In[118]:


telco_df['tenure_group'].value_counts()


# 5) Removing columns which isn't required

# In[119]:


telco_df.drop(['tenure','customerID'],axis=1,inplace=True)
telco_df.head()


# ## Data Exploration
# 
# 
# plot distribution of individual predictor by churn

# In[120]:


for j,predictor in enumerate(telco_df.drop(['Churn','MonthlyCharges','TotalCharges'],axis=1)):
    plt.figure(j)
    sns.countplot(data=telco_df,x=predictor)


# In[121]:


for i ,predictor in enumerate(telco_df.drop(['Churn','MonthlyCharges','TotalCharges'],axis=1)):
    plt.figure(i)
    sns.countplot(data=telco_df,x=predictor,hue='Churn')


# 

# 2) Convert target varibale "Churn" to binary numeric variable

# In[122]:


telco_df['Churn']=np.where(telco_df['Churn']=='Yes',1,0)


# In[123]:


telco_df['Churn'].value_counts()


# 3) Convert categorical variable into dummies variable

# In[124]:


data_dummies=pd.get_dummies(telco_df)


# In[125]:


data_dummies.head()


# In[126]:


data_dummies.shape


# now the no of predictors have been increased 

# 4) Let's find the relation between monthly charges and total charges

# In[127]:


sns.lmplot(data=data_dummies,x='MonthlyCharges',y='TotalCharges',fit_reg=False)


# As expected that if totalcharges increased monthly charges also increases.

# 5) Churn by monthly charges and total charges

# In[128]:


ax=sns.kdeplot(data_dummies.MonthlyCharges[(data_dummies['Churn']==0)],shade=True,color='purple')
ax=sns.kdeplot(data_dummies.MonthlyCharges[(data_dummies['Churn']==1)],shade=True,color='blue',ax=ax)
ax.set_xlabel('Monthly Charges')
ax.set_ylabel('Density')
ax.legend(['No churn','Churn'],loc='upper right')
ax.set_title('Monthly charges by churn')


# From above distribution we observed that as monthly charges increased their is hight chance that customer will churn.

# In[129]:


ax=sns.kdeplot(data_dummies.TotalCharges[(data_dummies['Churn']==0)],shade=True,color='purple')
ax=sns.kdeplot(data_dummies.TotalCharges[(data_dummies['Churn']==1)],shade=True,color='blue',ax=ax)
ax.set_xlabel('Total Charges')
ax.set_ylabel('Density')
ax.legend(['No churn','Churn'],loc='upper right')
ax.set_title('Total charges by churn')


# * Customer with low total charges are more likely to churn .
# 
# * Insights:The customer with low **total charges** ,**high monthly charges** and **tenure between 1-12** are more likely to be **churned**
# 

# 6) checking the outliers in continuos variable

# In[130]:


sns.boxplot(data_dummies['MonthlyCharges'])


# In[131]:


sns.boxplot(data_dummies['TotalCharges'])


# In[132]:


data_dummies[['MonthlyCharges','TotalCharges']].skew()


# TotalCharges is highly right skewed and it tells that 25% customers are pay more than USD 4000 total charges.

# ### Bivariate analysis

# In[133]:


sns.countplot(x='Partner',data=df,hue='Churn')


# In[134]:


non_churn=telco_df[telco_df['Churn']==0]
churn=telco_df[telco_df['Churn']==1]


# ### Feature selection for model building

# In[135]:


from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


# In[136]:


# object data
data=[i for i in telco_df if telco_df[i].dtype=='object']


# In[137]:


feature=telco_df[data]
feature.head()


# In[138]:


feature['gender']=np.where(feature['gender']=='Male',1,0)
feature['Partner']=np.where(feature['Partner']=='Yes',1,0)
feature['Dependents']=np.where(feature['Dependents']=='Yes',1,0)
feature['PhoneService']=np.where(feature['PhoneService']=='Yes',1,0)
feature['OnlineSecurity']=np.where(feature['OnlineSecurity']=='Yes',1,0)
feature['OnlineBackup']=np.where(feature['OnlineBackup']=='Yes',1,0)
feature['DeviceProtection']=np.where(feature['DeviceProtection']=='Yes',1,0)
feature['TechSupport']=np.where(feature['TechSupport']=='Yes',1,0)
feature['StreamingTV']=np.where(feature['StreamingTV']=='Yes',1,0)
feature['StreamingMovies']=np.where(feature['StreamingMovies']=='Yes',1,0)
feature['PaperlessBilling']=np.where(feature['PaperlessBilling']=='Yes',1,0)


# In[139]:


ordinal_label={k:i for i,k in enumerate(feature['Contract'].unique())}
feature['Contract']=feature['Contract'].map(ordinal_label)


# In[140]:


ordinal_label={k:i for i,k in enumerate(feature['PaymentMethod'].unique())}
feature['PaymentMethod']=feature['PaymentMethod'].map(ordinal_label)


# In[141]:


ordinal_label={k:i for i,k in enumerate(feature['MultipleLines'].unique())}
feature['MultipleLines']=feature['MultipleLines'].map(ordinal_label)


# In[142]:


ordinal_label={k:i for i,k in enumerate(feature['InternetService'].unique())}
feature['InternetService']=feature['InternetService'].map(ordinal_label)


# In[143]:


feature.head()


# In[144]:


feature_selec=pd.concat([feature,telco_df['Churn']],axis=1)


# In[145]:


feature_selec.head()


# In[146]:


X=feature_selec.drop(['Churn'],axis=1)
Y=feature_selec.Churn


# In[147]:


chi_p_value=chi2(X,Y)


# In[148]:


chi_p_value


# In[149]:


p_value=pd.DataFrame(chi_p_value[1])
p_value.index=feature.columns


# In[150]:


p_value.sort_index(ascending=False)


# Now we select the feature having p_value < 0.05 

# In[151]:


feature_selected=[i for i in p_value[0] if i<=0.05]


# In[152]:


feature_selected


# gender , phone service and multiple lines these 3 variables are not significant.so we will remove these features .

# In[153]:


sign_feat=feature_selec.drop(['gender','PhoneService','MultipleLines'],axis=1)


# In[154]:


sign_feat.columns


# In[155]:


# now we add the monthly and total charges in the data for model building


# In[156]:


DF=pd.concat([sign_feat,telco_df['MonthlyCharges'],telco_df['TotalCharges']],axis=1)
DF.head()


# In[ ]:





# In[157]:


DF.shape


# ### Model building

# In[158]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek


# In[159]:


# creating x and y 

X=DF.drop(['Churn'],axis=1)


# In[160]:


X.shape


# In[161]:


y=DF['Churn']
y


# In[162]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# ### Logistic regression

# In[163]:


from sklearn.linear_model import LogisticRegression


# In[164]:


lr=LogisticRegression()


# In[165]:


lr.fit(X_train,y_train)


# In[166]:


y_pred=lr.predict(X_test)


# In[167]:


y_pred


# In[168]:


print(confusion_matrix(y_test,y_pred))


# In[169]:


Accuracy=(916+188)/(916+188+122+181)
Accuracy


# In[170]:


print(classification_report(y_test,y_pred))


# * From report we can observed that for non churners precision and recall is high and for churners it's low because the model is   trained more on non churners customer than churners.
# 
# * Now we use upsampling to balance the data for better precision and recall
# 
# * For imbalance data accuracy is cursed.

# In[171]:


# handling imbalance data

sm=SMOTETomek(random_state=40)


# In[172]:


x_resamp,y_resamp=sm.fit_resample(X,y)


# In[173]:


y_resamp.value_counts()


# In[174]:


y.value_counts()


# In[175]:


xr_train,xr_test,yr_train,yr_test=train_test_split(x_resamp,y_resamp,test_size=0.2)


# In[176]:


LR=LogisticRegression()


# In[177]:


LR.fit(xr_train,yr_train)


# In[178]:


yr_pred=LR.predict(xr_test)


# In[179]:


print(classification_report(yr_test,yr_pred))


# * After handling imbalance data the precision and recall is good for both non churners and churners customers.
# 
# 
# 

# In[180]:


print(confusion_matrix(yr_test,yr_pred))


# * Here the 195 is false positive which indicates number of customers who actually non churners but model predicted are churners.It is also called as type I error.
#  
# * 146 is false negative which indicates number of customers who actually churned but model predicted are non churners.It is also called as type II error.
#  
# * here type I error is sensitive for business because the customers who are loyal to company are predicted as churners so we need to reduce this false positive rate but we cann't directly reduce the number because we need domain expertise who will decide the threshol value.

# In[181]:


Accuracy=(767+784)/(767+784+195+146)
Accuracy


# ### Random Forest

# In[182]:


from sklearn.ensemble import RandomForestClassifier


# In[183]:


RF=RandomForestClassifier()


# In[184]:


RF.fit(xr_train,yr_train)


# In[185]:


y_pred=RF.predict(xr_test)


# In[186]:


print(classification_report(yr_test,yr_pred))


# In[187]:


print(confusion_matrix(yr_test,yr_pred))


# In[188]:


## Hyperparameter tuning for finding best parameter

# grid search cv

# Number of trees in random forest
n_estimators=[60,100,120]

# maximum no of level in tree
max_depth=[2,8,None]


# number of feature to consider at every split

max_features=[0.2,0.6,1.0]

max_samples=[0.5,0.75,1.0]

# 108 different random forest train


# In[189]:


param_grid={'n_estimators':n_estimators,
             'max_depth':max_depth,
             'max_features':max_features,
              'max_samples':max_samples}

print(param_grid)


# In[190]:



from sklearn.model_selection import GridSearchCV


# In[191]:


grid=GridSearchCV(estimator=RF,param_grid=param_grid,cv=5,verbose=2,n_jobs=-1)


# In[192]:


grid.fit(xr_train,yr_train)


# In[193]:


grid.best_params_


# In[194]:


grid.best_score_


# In[195]:


yp=grid.predict(xr_test)


# In[196]:


yp


# In[197]:


print(classification_report(yr_test,yp))


# ### Model saving

# In[207]:


import pickle


# In[208]:


with open("grid_pkl",'wb') as f:
    pickle.dump(grid,f)


# In[209]:


with open("grid_pkl",'rb') as f:
    model=pickle.load(f)


# In[212]:


model.predict([[1,0,0,0,1,0,0,0,0,0,1,29.85,0,29.85]])


# In[ ]:





# In[ ]:




