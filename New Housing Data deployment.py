#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

housing = pd.read_csv('/Users/Tale/Desktop/housing-deployment-reg.csv')
housing.head()


# In[6]:


from sklearn.ensemble import RandomForestRegressor

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
r2_rf


# In[9]:


import pickle 

# Save the trained Random Forest model to a file with the specified name
pickle.dump(trained_pipe_rf, open('/Users/Tale/Desktop/house_prices_trained_pipe_streamlit.sav', 'wb'))


# In[ ]:




