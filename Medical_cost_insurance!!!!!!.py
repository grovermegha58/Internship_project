#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


# In[10]:


# Load the dataset
url = "https://github.com/FlipRoboTechnologies/ML-Datasets/raw/main/Medical%20Cost%20Insurance/medical_cost_insurance.csv"
df = pd.read_csv(url)


# In[11]:


# Check for missing values
print(df.isnull().sum())


# In[12]:


# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])


# In[13]:


# Split data into features and target variable
X = df.drop(columns=['charges'])
y = df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[14]:


# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)


# In[15]:


# Prediction
new_data = pd.DataFrame({
    'age': [35],
    'sex': ['male'],  # encoded value: 1 for male
    'bmi': [25.5],
    'children': [2],
    'smoker': ['no'],  # encoded value: 0 for no
    'region': ['southwest']  # encoded value: 3 for southwest
})

# Encode categorical variables in new data using the same LabelEncoders
new_data['sex'] = le_sex.transform(new_data['sex'])
new_data['smoker'] = le_smoker.transform(new_data['smoker'])
new_data['region'] = le_region.transform(new_data['region'])

# Make prediction
prediction = model.predict(new_data)
print("Predicted insurance cost:", prediction[0])


# In[ ]:




