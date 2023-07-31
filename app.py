#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle

# Load the housing data
housing = pd.read_csv('/Users/Tale/Desktop/housing-deployment-reg.csv')

# Define X and y
X = housing.drop(columns=['SalePrice'])
y = housing['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Define the pipeline
pipe_rf = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestRegressor(random_state=8)
)

# Define the parameter grid for the pipeline
pipe_params_rf = {
    'simpleimputer__strategy': ['median'],
    'randomforestregressor__n_estimators': [100, 200, 300]
}

# Perform grid search to find the best parameters
trained_pipe_rf = GridSearchCV(pipe_rf, pipe_params_rf, cv=5)
trained_pipe_rf.fit(X_train, y_train)

# Make predictions on the test set and calculate the R2 score
y_pred_rf = trained_pipe_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Save the trained Random Forest model to a file
with open('house_prices_trained_pipe_streamlit.sav', 'wb') as file:
    pickle.dump(trained_pipe_rf, file)

# Create the Streamlit app
def main():
    st.title('House Prices Prediction App')

    # User input for features
    st.header('Enter the features of the house:')
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Make prediction using the trained model


# In[ ]:




