#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv ('desktop\data\loan-pyhton-project\loan.csv')


# In[2]:


df.head()


# In[3]:


df.info()


# In[4]:


df.nunique()


# In[5]:


import seaborn as sns
sns.catplot(x="Loan_Status",kind="count", data=df)


# In[6]:


df.corr()


# In[7]:


sns.catplot(x="Gender" ,hue="Loan_Status",kind="count", data=df)


# In[8]:


col=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"]
for i in col:
    sns.catplot(x=i ,hue="Loan_Status",kind="count", data=df)


# In[10]:


cols=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
for i in cols:
    sns.displot(df, x=i,hue="Loan_Status", kde=True)

We can see here that the most important numerical features is:  "CoapplicantIncome"
# In[11]:


colmn=["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area","CoapplicantIncome"]
data= df[colmn]


# In[12]:


data['Dependents'] = data['Dependents'].replace(['3+'],3)
data['Dependents'] = data['Dependents'].replace(['2'],2)
data['Dependents'] = data['Dependents'].replace(['1'],1)
data['Dependents'] = data['Dependents'].replace(['0'],0)


# In[13]:


data.loc[data['Loan_Status'] == 'Y', 'Loan_Status'] = 1
data.loc[data['Loan_Status'] == 'N', 'Loan_Status'] = 0


# In[14]:


data


# In[15]:


from sklearn import preprocessing
colenc=["Gender","Married","Education","Self_Employed"]
for i in colenc:
    le = preprocessing.LabelEncoder()
    le.fit(data[i])
    data[i]=le.transform(data[i])


# In[16]:


data.head()


# In[17]:


y=data["Loan_Status"]
x=data.drop("Loan_Status",axis=1)


# In[18]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
one_hot = pd.get_dummies(x["Property_Area"])
x = x.drop("Property_Area",axis = 1)
x = x.join(one_hot)


# In[19]:


x.head()


# In[20]:


import numpy as np
colmns=["Gender","Married","Dependents","Education","Self_Employed","Credit_History","CoapplicantIncome"]
for i in colmns:
    x[i] = x[i].replace(np.nan, x[i].mean())


# In[21]:


y=y.astype('int') 


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=42)


# In[23]:


from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train ,y_train)
y_pred=model.predict(x_test)
y_predtrain=model.predict(x_train)
print('the accuracy for the test data is',accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import accuracy_score
model2 = RandomForestClassifier(max_depth=2, random_state=0)
model2.fit(x_train, y_train)
from sklearn.metrics import confusion_matrix
y_pred=model2.predict(x_test)
y_predtrain=model2.predict(x_train)
print('the accuracy for the test data is',accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

a result very similar for randomforest algorithm and logistic regression algorithm