#!/usr/bin/env python
# coding: utf-8

# # Reading and Analyzing Bank Customer Churn Data.¶
# Before we start our analysis, first, we need to view the dataset. It is essential to view the data and check the columns. Let's take a look.

# In[2]:


#--- Import Pandas ---
import pandas as pd
#--- Read in dataset ----
df = pd.read_csv('churn.csv')

df


# # Counting Null Values in the Data.
# Great!!! We have our dataset loaded. Now, we need to ensure that the dataset does not contain any null values. So, let's check for null values in our dataset.

# In[3]:


# Check for missing (null) values in the DataFrame.
# The 'isnull()' method returns a DataFrame of the same shape as 'df',
# with True for each null (missing) value and False otherwise.
# The 'sum()' method then calculates the total number of null values in each column.
null_values = df.isnull().sum()
print(null_values)


# # Counting Duplicates.
# Wow!!! Our dataset does not have any null values. Now, we need to ensure that the data does not contain any duplicates. Checking for duplicate rows is crucial for maintaining data accuracy. Let's go ahead and check for them.

# In[4]:


# Check for duplicate rows in the DataFrame.
# The 'duplicated()' method returns a Series with True for each duplicate row
# and False for unique rows. 
# The 'sum()' method counts the total number of duplicate rows in the DataFrame.
duplicates = df.duplicated().sum()

print(duplicates)


# # Exited Customer Distribution Analysis.¶
# Incredible!!! We have successfully verified that there are no duplicates. Now, we need to check the distribution of 'Exited' customers in our dataset. Let's proceed to check it.
# The column 'Exited' in the dataset likely contains categorical data (e.g., 1 for exited, 0 for not exited).
# Our goal is to examine the distribution of these values to gain insights, such as the proportion of customers who have churned compared to those who remain.
# 
# Understanding the distribution helps to assess the balance of the dataset. A highly imbalanced dataset (e.g., very few customers exited compared to those who stayed) might require special handling during analysis or model building, such as resampling techniques.
# By proceeding to check the distribution, you'll gain a clearer picture of customer churn in your dataset, which is crucial for further analysis or predictive modeling.
# 

# In[5]:


# Count the occurrences of each unique value in the 'Exited' column.
# The 'value_counts()' method returns a Series with the counts of each unique value,
# providing the distribution of customers who have exited (e.g., churned) versus those who have not.
values = df['Exited'].value_counts()
print(values)


# In[7]:


import matplotlib.pyplot as plt

# Count the occurrences of each unique value in the 'Exited' column
values = df['Exited'].value_counts()

# Plot a bar graph
plt.figure(figsize=(8, 6))
values.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Exited Customers', fontsize=16)
plt.xlabel('Exited (0 = Not Exited, 1 = Exited)', fontsize=14)
plt.ylabel('Number of Customers', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # Removing Unnecessary Columns.
# Now, we need to drop irrelevant columns. This action will enhance the clarity and focus of our data analysis process. Let's work on it.

# In[8]:


df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1) # rownumber, customer id and Lastname is not a feature column and hence we can drop them

#--- Inspect data ---
df


# # Categorizing Dataset Columns.
# We've successfully removed the columns. Now, we're categorizing the columns into numerical and nominal types. This categorization facilitates targeted exploration and manipulation of data based on their respective natures. Let's take a look.

# In[10]:


numbCol = ['EstimatedSalary', 'Balance', 'CreditScore', 'Age'] #Define a list named 'numbCol' with numeric columns: 'EstimatedSalary', 'Balance', 'CreditScore', and 'Age'.
nomCol = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender', 'NumOfProducts', 'Tenure'] #Define a list named nomCol with nominal categorical columns: 'HasCrCard', 'IsActiveMember', 'Geography', 'Gender', 'NumOfProducts', and 'Tenure'.
numbCol, nomCol


# # Outlier Detection Analysis.
# We have categorized the columns. Now, we aim to identify outliers within our dataset. By doing this, we can facilitate robust data cleansing and analysis strategies. So, let's check it.

# In[12]:


pip install numpy==1.22.0


# In[13]:


from scipy import stats

# Define the threshold for outlier detection
threshold = 3

# Create a dictionary to store the count of outliers for each column
outlier_counts = {}

# Iterate over each numerical column
for col in numbCol:
    # Calculate Z-scores for the current column
    z_scores = stats.zscore(df[col])
    
    # Find outliers (absolute Z-scores greater than the threshold)
    outliers = (abs(z_scores) > threshold)
    
    # Count the number of outliers for the current column
    outlier_counts[col] = outliers.sum()

# outlier_counts now contains the count of outliers for each column
outlier_counts


# # Data Integrity Check.
# Ohhhhh!!! We've found outliers in two columns. Before removing them, we need to ensure that it's okay to drop them. Because age can be any number, although we set one limit and check for real outliers. This examination is crucial for maintaining data integrity and reliability in subsequent analyses. Let's take a look.

# In[15]:


# Filter the DataFrame for values in the "Age" column greater than 100 and less than 1
ages_check = df[(df['Age'] > 100) | (df['Age'] < 1)].shape[0]
# Filter the DataFrame for values in the "Age" column greater than 100 and less than 300
credit_check = df[(df['CreditScore'] > 900) | (df['CreditScore'] < 300)].shape[0]
ages_check , credit_check


# # Converting Estimated Salary and Balance to Integer Data Type.
# Fantastic!!! There are no real outliers in our columns. Now, we need to change the columns from float to integer values. By doing this, we gain a better understanding from a broader perspective. Let's convert them.

# In[18]:


df.dtypes


# In[19]:


df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)
df['Balance'] = df['Balance'].astype(int)

df.dtypes


# # Label Encoding for Categorical Data.
# We've changed the columns successfully. Now, we need to change the words of the categories into numbers. This helps us to better understand and analyze the data. Let's do it.

# In[21]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'Geography' column
df['Geography'] = label_encoder.fit_transform(df['Geography'])

# Encode the 'Gender' column
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.dtypes


# # Percentage of Churned Customers based on Credit Card Ownership.
# We've successfully completed the conversion. Now, we need to check how the column relates to the 'Exited' column. This analysis helps us understand how credit card services affect customer retention strategies. Let's get started.

# In[22]:


credit_churn_percentage = df.groupby('HasCrCard')['Exited'].mean() * 100

print(credit_churn_percentage)


# In[23]:


import matplotlib.pyplot as plt

# Calculate the churn percentage grouped by 'HasCrCard'
credit_churn_percentage = df.groupby('HasCrCard')['Exited'].mean() * 100

# Print the churn percentage
print(credit_churn_percentage)

# Plot a bar graph
plt.figure(figsize=(8, 6))
credit_churn_percentage.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Customer Churn Percentage by Credit Card Ownership', fontsize=16)
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Churn Percentage (%)', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[24]:


#We found out that credit card services have no impact on customer retention. Let's go ahead and drop the column.
df = df.drop('HasCrCard', axis=1)

#--- Inspect data ---
df


# # Standardizing Numerical Data with StandardScaler.
# We have successfully removed the column. Now, we need to standardize numerical data within our dataset. This preprocessing step enhances the consistency and comparability of numerical attributes, making modeling and analysis processes more effective downstream.

# In[25]:


from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the columns in numbCol and transform them
df[numbCol] = scaler.fit_transform(df[numbCol])


# # Splitting Data into Features and Target Variable.
# We have successfully standardized the data. Now, we need to split it. This division allows for distinct handling of predictors and the variable to be predicted, making supervised learning tasks easier. Let's get started.

# In[26]:


# Create a new DataFrame x1 by dropping the 'Exited' column
x1 = df.drop('Exited', axis=1)

# Extract the 'Exited' column into a Series y1
y1 = df['Exited']
x1, y1


# # Oversampling Minority Class with SMOTE.
# We have successfully divided the data. Now, we need to oversample the data to handle the imbalance. By doing this, we'll increase the representation of the minority class, ensuring a balanced distribution of classes. Let's get started.
# 
# Oversampling the Minority Class with SMOTE (Synthetic Minority Over-sampling Technique) is a technique used to address the issue of class imbalance in machine learning datasets. When the number of samples in one class (often the minority class) is significantly lower than in the other class (majority class), it can lead to biased models that perform poorly on the minority class. SMOTE helps to balance the classes by generating synthetic data points.
# 
# 

# In[28]:


pip install imbalanced-learn


# In[29]:


from imblearn.over_sampling import SMOTE

# Initialize SMOTE with a sampling strategy of 1 (equal class distribution)
smote = SMOTE(sampling_strategy=1)

# Apply fit_resample to x1 and y1, converting results to NumPy arrays
x1_resampled, y1_resampled = smote.fit_resample(x1, y1)

# Convert the resampled features and target variable to NumPy arrays
x1_resampled = x1_resampled.to_numpy()
y1_resampled = y1_resampled.to_numpy()


# # Splitting Oversampled Data for Training and Testing.
# We have balanced the data. Now, we need to split it into training and testing sets. This division ensures an independent dataset for model validation, which facilitates accurate performance assessment and estimation of generalization capability. Let's proceed.

# In[30]:


from sklearn.model_selection import train_test_split

# Split the resampled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x1_resampled, y1_resampled, test_size=0.2, random_state=42)


# # Model Evaluation
# We have split the data. In this section, we are implementing a program to evaluate the performance of three different machine learning models: Support Vector Machine (SVM), Random Forest, and Logistic Regression. The goal is to train each model on the training data, make predictions on the test data, and then evaluate the accuracy of each model.

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test)

# Evaluate the model's performance using accuracy_score
accuracy1 = accuracy_score(y_test, y_pred)
print(accuracy1)


# In[34]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the SVM classifier with a linear kernel
svm_model = SVC(kernel='linear')

# Train the SVM classifier on the training data
svm_model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred_svm = svm_model.predict(x_test)

# Evaluate the model's accuracy using accuracy_score
accuracy2 = accuracy_score(y_test, y_pred_svm)

accuracy2


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the RandomForestClassifier model
rf_model = RandomForestClassifier()

# Train the RandomForestClassifier model on the training data
rf_model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred_rf = rf_model.predict(x_test)

# Evaluate the model's accuracy using accuracy_score
accuracy3 = accuracy_score(y_test, y_pred_rf)

accuracy3


# # Evaluation Metrics: Precision, Recall, and F1 Score.
# Since we found better accuracy with the Random Forest Classifier model among the three models we considered, let's proceed to evaluate its performance using other metrics.

# In[37]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
precision, recall, f1


# # Hyperparameter Tuning with Randomized Search: Random Forest.
# We've obtained the remaining metrics. Now, let's fine-tune the model to improve its performance. Let's get started!

# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the Random Forest model (rf_Model)
rf_Model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_Model, param_distributions=param_grid, 
                                   cv=5, verbose=0, n_jobs=1, random_state=42)

# Fit the model
random_search.fit(x_train, y_train)

# Make predictions
y_pred1 = random_search.best_estimator_.predict(x_test)

# Calculate metrics
accuracy4 = accuracy_score(y_test, y_pred1)
precision4 = precision_score(y_test, y_pred1)
recall4 = recall_score(y_test, y_pred1)
f1_4 = f1_score(y_test, y_pred1)

accuracy4, precision4,recall4, f1_4


# # Feature Importance Analysis: Random Forest Model.
# We've fine-tuned the model. Now, let's determine the importance of the columns in our dataset. Let's proceed to do that.

# In[41]:


import pandas as pd

# Assuming the RandomizedSearchCV object is named 'random_search'
# Retrieve the best estimator from RandomizedSearchCV
best_rf_model = random_search.best_estimator_

# Extract feature importances from the best model
feature_importances = best_rf_model.feature_importances_

# Get the feature names from the DataFrame (excluding 'Exited' column)
feature_names = df.drop('Exited', axis=1).columns

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df


# # Loading Holdout Dataset for Churn Prediction.
# We've moved to the final steps. Before we start our prediction, let's first view the holdout dataset. It's essential to inspect the data and check the columns. Let's take a look.
# 
# Using a holdout dataset for churn prediction is crucial for evaluating the performance of your model in a real-world scenario. The holdout dataset acts as a "test" set that helps you assess how well your model generalizes to unseen data, which is essential for avoiding overfitting and ensuring the model's reliability.
# 
# By inspecting the dataset, you ensure that the model will have a fair and realistic test of its prediction capabilities.

# In[50]:


# Load the Holdout Dataset

holdout_df = pd.read_csv('holdout_churn.csv')
holdout_df


# # Preprocessing Holdout Data for Churn Prediction.
# Great! We have our dataset loaded. Now, let's process it before making predictions. So, let's check for null values in our dataset, converting to integers, apply label encoding and perform the standard Scaling and tranformation.

# In[51]:


from sklearn.preprocessing import LabelEncoder

# Drop columns: 'HasCrCard', 'RowNumber', 'CustomerId', and 'Surname'
holdout_df = holdout_df.drop(columns=['HasCrCard', 'RowNumber', 'CustomerId', 'Surname'])

# Convert the 'EstimatedSalary' column to integer data type
holdout_df['EstimatedSalary'] = holdout_df['EstimatedSalary'].astype(int)

# Convert the 'Balance' column to integer data type
holdout_df['Balance'] = holdout_df['Balance'].astype(int)

# Perform label encoding for each feature in the list lstforle
le = LabelEncoder()
lstforle = ['Geography', 'Gender']
for feature in lstforle:
    holdout_df[feature] = le.fit_transform(holdout_df[feature])
    

# Use transform() function from the sc object to transform numerical columns in the holdout DataFrame
holdout_df[numbCol] = sc.transform(holdout_df[numbCol])


# # Making Predictions on Holdout Data.
# After preprocessing the holdout dataset with StandardScaler to ensure consistency in feature scaling, we utilize the trained Random Forest model to make predictions on the preprocessed holdout data. This step represents a crucial aspect of model evaluation and deployment, as it assesses the model's performance on unseen data, validating its effectiveness in real-world scenarios. Let's work on it.

# In[52]:


# Make predictions using predict() function from rf_RandomGrid on the values of the holdout DataFrame
predictions = rf_RandomGrid.predict(holdout_df.values)
predictions


# # Loading Original Churn Results.
# Now, let's view the real values for the holdout data. It's essential to inspect the data and check the values. Let's take a look.

# In[54]:


result_df = pd.read_csv('holdout_churn_result.csv')
result_df


# # Evaluating Predictive Model Performance with F1 Score.
# Now we aim to evaluate the predicted result with the original values. By doing this, we can assess the performance of the model. Let's check it.

# In[55]:


# Calculate the F1 score using the f1_score() function
f1_score = f1_score(result_df['Exited'].values, predictions)
f1_score


# # Key Takeaway:
# The F1 score provides a useful single metric to evaluate model performance, especially for imbalanced datasets. While a score of 0.457 indicates moderate performance, it highlights the need for further refinement to improve predictions.

# In[ ]:




