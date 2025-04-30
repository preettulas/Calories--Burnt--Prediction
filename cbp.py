#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# In[38]:


pip install xgboost


# ### DATA COLLECTION AND PROCESSING

# In[39]:


#LOADING THE DATA FROM CSV FILE TO A PANDAS DATAFRAME


# In[40]:


calories = pd.read_csv('calories.csv')


# In[41]:


#print the first five rows of the dataframe
calories.head()


# In[42]:


exercise_data = pd.read_csv('exercise.csv')


# In[43]:


exercise_data.head()


# ### COMBINING THE TWO DATAFRAMES

# In[44]:


calories_data = pd.concat([exercise_data, calories['Calories']],axis=1)


# In[45]:


calories_data.head()


# In[46]:


#CHECKING THE NUMBER OF ROWS AND COLUMNS
calories_data.shape


# In[47]:


#GETTING SOME INFORMATION ABOUT THE DATA
calories_data.info()


# In[48]:


#CHECKING FOR MISSING VALUE
calories_data.isnull().sum()


# ### DATA ANALYSIS

# In[49]:


# get some statistical measures about the data
calories_data.describe()


# ### DATA VISUALIZATION

# In[50]:


sns.set()


# In[51]:


#finding the distribution of "age column"
sns.distplot(calories_data['Age'])


# In[52]:


sns.distplot(calories_data['Height'])


# In[53]:


sns.distplot(calories_data['Weight'])


# In[54]:


sns.distplot(calories_data['Duration'])


# In[55]:


sns.distplot(calories_data['Heart_Rate'])


# In[56]:


sns.distplot(calories_data['Body_Temp'])


# In[57]:


sns.distplot(calories_data['Calories'])


# ### FINDING THE CORRELATION IN THE DATASET

# ### 1.positive correlation
# 2.negative correlation

# In[58]:


data = np.random.rand(10, 10)
correlation = pd.DataFrame(data).corr()


# In[59]:


plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()


# ### CONVERING THE NEXT DATA TO NUMERICAL VALUES

# In[60]:


calories_data.replace({"Gender":{'male':0, 'female':1}}, inplace=True)


# In[61]:


calories_data.head()


# ### SEPERATING FEATURES AND TARGET

# In[62]:


x= calories_data.drop(columns=['User_ID','Calories'], axis=1)
y= calories_data['Calories']


# In[63]:


print(x)


# In[64]:


print(y)


# ### SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

# In[65]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# In[66]:


print(x.shape, x_train.shape, x_test.shape)


# ### MODEL TRAINING

# In[67]:


model = XGBRegressor()


# In[68]:


#training the model with x_train
model.fit(x_train, y_train)


# ### EVALUATION

# ### PREDICTION ON TEST DATA

# In[69]:


test_data_prediction = model.predict(x_test)


# In[70]:


print(test_data_prediction)


# ### MEAN ABSOLUTE ERROR

# In[71]:


mae = metrics.mean_absolute_error(y_test, test_data_prediction)


# In[72]:


print("Mean Absolute Error =",mae)


# In[ ]:





# In[ ]:





# In[ ]:
