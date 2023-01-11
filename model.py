#importing the required packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,accuracy_score
import pickle


# In[2]:


#loading the given csv data
data = pd.read_csv(r'C:\Users\Dell\Downloads\cardio_data1.csv')
data.head()


# In[3]:


data.columns


# In[4]:


#finding the number of columns and rows
data.shape


# In[5]:


#Statistical summary
data.describe()


# In[6]:


#finding the data information
data.info()





# In[9]:


data["cardio"].value_counts(normalize=True)



# In[11]:


data["smoke"].value_counts(normalize=True)




# In[18]:


#age is given in number of days.So we want to change into year
#data["age"]=np.round(data["age"]/365)
data['age']=np.round(data['age']/365)


# In[19]:

# In[21]:


#Missing Values


# In[22]:


#Checking for null values
data.isna().sum()


# In[23]:


#Since the alco and chlesterol are categorical values,null values can be filled with mode.


# In[24]:


cols=['alco','cholesterol','gender']
for i in cols:
    data[i]=data[i].fillna(data[i].mode()[0])
    


# In[25]:


#plotting of missing values of height


# In[26]:


sns.histplot(data['height'],binwidth=5)
plt.title('Distribution of height')
plt.show


# In[27]:


#Since the data distribution is normal,we can fill the missing values of height with mean


# In[28]:


data['height']=data['height'].fillna(data['height'].mean())


# In[29]:


data.isna().sum()


# In[30]:


#finding unique values
data.nunique()


# In[31]:


#Dropping of 'id' column


# In[32]:


data.drop('id',axis=1,inplace=True)


# In[33]:


data.head()


# In[34]:


#sorting categorical values
categorical_features = data.select_dtypes(include=[np.object])
categorical_features.head()


# In[35]:


#sorting numerical values
numerical_features = data.select_dtypes(include=[np.number])
numerical_features.head()


# In[36]:


numerical_features.columns


# #### Outlier detection

# #### age

# In[37]:


#Managing Outliers
Q1 = np.percentile(data['age'],25,interpolation='midpoint')
Q2 = np.percentile(data['age'],50,interpolation='midpoint')
Q3 = np.percentile(data['age'],75,interpolation='midpoint')
IQR = Q3-Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR
print(low_lim)
print(up_lim)


# In[38]:


outlier = []
for x in data ['age']:
    if((x > up_lim) or (x < low_lim)):
        outlier.append(x)
outlier


# #### height

# In[39]:


#Managing Outliers
Q1 = np.percentile(data['height'],25,interpolation='midpoint')
Q2 = np.percentile(data['height'],50,interpolation='midpoint')
Q3 = np.percentile(data['height'],75,interpolation='midpoint')
IQR = Q3-Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR
print(low_lim)
print(up_lim)


# In[40]:


outlier = []
for x in data ['height']:
    if((x > up_lim) or (x < low_lim)):
        outlier.append(x)
outlier
len(outlier)


# In[41]:


data[data['height'] < 125]


# In[42]:


data=data[data['height']>= 125]


# In[43]:


data[data['height'] > 200]


# In[44]:


data.drop(index=6486,axis=1,inplace=True)


# #### weight

# In[45]:


#Managing Outliers
Q1 = np.percentile(data['weight'],25,interpolation='midpoint')
Q2 = np.percentile(data['weight'],50,interpolation='midpoint')
Q3 = np.percentile(data['weight'],75,interpolation='midpoint')
IQR = Q3-Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR
print(low_lim)
print(up_lim)


# In[46]:


outlier = []
for x in data ['weight']:
    if((x > up_lim) or (x < low_lim)):
        outlier.append(x)
outlier
len(outlier)


# In[47]:


data=data[data['weight']> 40]


# #### ap_hi

# In[48]:


#Managing Outliers
Q1 = np.percentile(data['ap_hi'],25,interpolation='midpoint')
Q2 = np.percentile(data['ap_hi'],50,interpolation='midpoint')
Q3 = np.percentile(data['ap_hi'],75,interpolation='midpoint')
IQR = Q3-Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR
print(low_lim)
print(up_lim)


# In[49]:


outlier = []
for x in data ['ap_hi']:
    if((x > up_lim) or (x < low_lim)):
        outlier.append(x)
outlier
len(outlier)


# #### ap_lo

# In[50]:


#Managing Outliers
Q1 = np.percentile(data['ap_lo'],25,interpolation='midpoint')
Q2 = np.percentile(data['ap_lo'],50,interpolation='midpoint')
Q3 = np.percentile(data['ap_lo'],75,interpolation='midpoint')
IQR = Q3-Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR
print(low_lim)
print(up_lim)


# In[51]:


outlier = []
for x in data ['ap_lo']:
    if((x > up_lim) or (x < low_lim)):
        outlier.append(x)
outlier
len(outlier)


# In[52]:


## systolic pressure below 90 mmHg and above 180mmHg needs medical intervention
critical_aphi = (data.ap_hi <70) | (data.ap_hi > 200)
data[critical_aphi].ap_hi.count() / data.ap_hi.count()


# In[53]:


#dropping 0.4% data
data.drop(data[critical_aphi].index, inplace=True)


# In[54]:


## diastolic pressure below 60mmHg ang above 110mmHg is dangerous
critical_aplo = (data.ap_lo <40) | (data.ap_lo > 130)
data[critical_aplo].ap_lo.count() / data.ap_lo.count()


# In[55]:


#dropping 1.49% data
data.drop(data[critical_aplo].index, inplace=True)


# In[56]:


#plotting api_hi and api_lo
plt.style.use('bmh')
sns.scatterplot(x = 'ap_hi', y = 'ap_lo' , hue= 'cardio' , data = data,  s=40)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()


# ### BMI analysis

# In[57]:


data_bmi = data.copy()


# In[58]:


sns.scatterplot(x ='height', y ='weight', hue = 'cardio', data= data_bmi);


# In[59]:


#create bmi feature
def BMI(data):
    return round(data['weight'] / (data ['height'] /100)**2,2)
data_bmi['BMI'] = data_bmi.apply(BMI, axis=1)


# In[60]:


data_bmi.describe().T


# In[61]:


data_num = data.select_dtypes(include=[np.number])
data_num.columns


# ### Encoding

# In[62]:


#categorical_features.columns


# In[63]:


#apply label encoding
from sklearn.preprocessing import LabelEncoder
columns =['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
label_encoder = LabelEncoder()
for i in columns:
    data[i]=label_encoder.fit_transform(data[i])
data.head()


# In[64]:


#plot correlation matrix
corrmatrix = data.corr()
plt.subplots(figsize=(20,8))
sns.heatmap(corrmatrix,annot=True,cmap='YlGnBu');


# In[65]:


data.dtypes


# In[66]:


data.columns


# In[67]:


#converting to dataframe
data_cat = data.drop(data_num,axis=1)
data_cat = pd.DataFrame(data_cat)


# In[68]:


data_cat.columns


# In[69]:


data_num.columns


# In[70]:


data_num.reset_index(drop=True,inplace=True)
data_cat.reset_index(drop=True,inplace=True)


# In[71]:


#concat numerical and categorical columns
data = pd.concat([data_num,data_cat],axis=1)
data.head()


# In[72]:


data.isna().sum()


# In[73]:


#Standard Scaling


# In[74]:


#std_sclr=StandardScaler()


# In[75]:


#data_num=std_sclr.fit_transform(data_num)


# In[76]:


#data_num=pd.DataFrame(data_num)


# In[77]:


#data_num.columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#data_num.describe().T


# In[78]:


#concat numerical and categorical columns
#data1 = pd.concat([data_num,data_cat],axis=1)
#data1.head()


# In[79]:


X=data.drop(['cardio','alco'],axis=1)
Y=data['cardio']


# In[80]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[81]:
X_train




# In[82]:
from sklearn.linear_model import LogisticRegression

LR3=LogisticRegression()
LR_model3=LR3.fit(X_train,Y_train)
prediction3=LR_model3.predict(X_test)


# In[83]:

print("The precision score is\t :",precision_score(Y_test,prediction3))
print("The accuracy score is\t :",accuracy_score(Y_test,prediction3))
print("The recall score is\t :",recall_score(Y_test,prediction3))


# In[84]:

filename='model.pkl'
pickle.dump(LR3,open(filename,'wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
X_test.head()

# In[85]:


print(model.predict([[52.0,165.0,64.0,130,70,0,2,1,0,0]]))


# In[86]:


print(model.predict([[-0.491572,0.452568,-0.853208,-1.002751,-0.139432,1,1,1,0,1]]))


# In[ ]:




