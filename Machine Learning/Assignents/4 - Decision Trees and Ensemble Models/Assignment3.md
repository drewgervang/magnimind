Step 1:
Read the test and train datasets
Separate the SalePrice as target variable
Create a train and validation dataset from the train dataset that you created in the first step
Check if you have any categorical features
Check if you have any numerical features
​
Step 2:
Create a pipeline of SimpleImputer and StandardScaler transformers for the numerical data
Create a SimpleImputer and OneHotEncoder for the categorical data
Bundle the pre-processing steps into a column transformer
Create a RandomForestRegressor