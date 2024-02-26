# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
'''STEP 0 - Import libraries: 
We started by importing the necessary Python libraries that we'll use 
throughout the project. numpy and pandas are for data manipulation, and load_iris from sklearn.datasets is to load 
the dataset.'''

'''
Step 1.1: Data Loading and Initial Exploration

Load the dataset: The Iris dataset was loaded using load_iris(). This dataset includes 150 samples of iris flowers, 
with features like sepal length, sepal width, petal length, and petal width, along with the species of iris (the 
target variable).
'''

# Load the dataset
iris = load_iris()

'''
Step 1.2: DataFraming

Create a DataFrame: We then created a Pandas DataFrame from the dataset. A DataFrame is a 2-dimensional labeled data 
structure with columns of potentially different types. This makes it easy to work with structured data. We combined 
the features and the target variable into one DataFrame for convenience.
'''

# iris['data']: This part extracts the feature data from the Iris dataset.
# The features include measurements such as sepal length, sepal width, petal length, and petal width for each flower.
features_data = iris['data']

# iris['target']: This extracts the target data from the Iris dataset.
# The target is the species of each flower, encoded as integers (0, 1, 2).
target_data = iris['target']

# np.c_[iris['data'], iris['target']]:
# The np.c_ is a NumPy function that concatenates the feature data and target data along the second axis (column-wise).
# This effectively adds the target data as the last column of the feature data.
combined_data = np.c_[features_data, target_data]

# Create a list of column names
column_names = iris['feature_names'] + ['target']

# This constructs a Pandas DataFrame using the concatenated data
# as the data source and the list of column names as the column headers.
df = pd.DataFrame(data=combined_data, columns=column_names)


# Display the first few rows of the dataframe
print(df.head())

'''
Step 2: Data Cleaning and Preprocessing 

Although the Iris dataset is relatively clean, let's pretend we need to 
handle some missing data for the sake of practice. We'll artificially introduce some missing values and then handle 
them.
'''

'''df.iloc[...] = np.nan: This line uses .iloc to access specific rows and columns in the DataFrame by their integer 
index positions. It sets the values in rows 10 through 19 (inclusive of 10 and exclusive of 20) of the third column (
column_to_modify = 2) to np.nan, which stands for "Not a Number" and is used by pandas to represent missing values.'''
# Select a subset of the DataFrame to introduce missing values
rows_to_modify = slice(10, 20)  # Specifies the rows 10 through 19
column_to_modify = 2  # Specifies the third column (indexing starts at 0)

# Introduce missing values (NaN) into the specified rows and column
df.iloc[rows_to_modify, column_to_modify] = np.nan

'''df.mean(): This calculates the mean of each column in the DataFrame, ignoring NaN (missing) values. It returns a 
Series where the index labels correspond to the column names. 

df.fillna(column_means, inplace=True): This replaces 
all NaN values in the DataFrame with the mean value of their respective columns. The inplace=True argument modifies 
the original DataFrame directly, saving the need to assign the result back to df.'''

# Handle missing values by replacing them with the mean of the column
# Calculate the mean of each column, ignoring NaN values
column_means = df.mean()

# Replace NaN values in the DataFrame with the mean of their respective columns
df.fillna(column_means, inplace=True)

'''df.isnull(): This generates a boolean mask where True indicates the presence of a NaN value and False indicates a 
non-missing value. .sum(): When applied to the boolean mask, this counts the number of True values in each column, 
effectively giving us the count of NaN (missing) values in each column.'''

# Check if there are any missing values left
# Check each column for the presence of NaN values and sum them up
missing_values_count = df.isnull().sum()

# Print the count of NaN values in each column
print(missing_values_count)

'''Step 3: Feature Selection 

For simplicity, let's use all the features available in the Iris dataset for our model. 
In practice, feature selection might involve more sophisticated techniques to choose the most relevant features for 
your model.'''