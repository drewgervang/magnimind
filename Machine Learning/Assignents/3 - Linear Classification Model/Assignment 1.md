Assignment 1 & Submit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
​
Steps:
Import digits data set from Scikitlearn datasets library. Use load_digits(). When loaded, the dataset comes with data and target values.
Assign data to X and target to y
Check the shape of the data
Use np.bincount to print the number of uniqe elements of the target vriable y
Split data into train and test datasets. Use stratification when splitting. You can set your random_state to 42
Normalize your dataset. When normalizing, simply divide your dataset by the maximum of the train dataset. To find the maximum, use `max(
