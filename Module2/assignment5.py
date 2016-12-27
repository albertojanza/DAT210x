import pandas as pd
import numpy as np


#
# TODO:
# Load up the dataset, setting correct header labels.
#
# .. your code here ..
df = pd.read_csv('Module2/Datasets/census.data', index_col = 0, names = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'])


#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#
# .. your code here ..
df.head()
df.dtypes
df['capital-gain'].unique()
df = pd.read_csv('Module2/Datasets/census.data', index_col = 0, na_values = '?',
                 names = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'])
df['capital-gain'].unique()
df.dtypes

#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). Think to yourself, does it generally
# make more sense to have a numeric type or a series of categories
# for these somewhat ambigious features?
#
# .. your code here ..
#
df['education'].unique() #Categorial ordinal
df['sex'].unique() #Categorical nominal
df['classification'].unique() #Categorical ordinal (nly two values)
ordered_education = [ 'Preschool','1st-4th', '5th-6th', '7th-8th','9th','10th','11th','12th','HS-grad',  'Some-college', 'Bachelors',  'Masters',  'Doctorate'] 
df.education = df.education.astype("category", ordered=True, categories=ordered_education).cat.codes

ordered_classification = ['<=50K', '>50K']
df.classification = df.classification.astype("category", ordered=True, categories=ordered_classification).cat.codes

ordered_sex = ['Male', 'Female']
df.sex = df.sex.astype("category").cat.codes
df = pd.get_dummies(df,columns=['sex'])


#
# TODO:
# Print out your dataframe
#
# .. your code here ..

df.head()
#   education  age  capital-gain   race  capital-loss  hours-per-week  \
#0         10   39        2174.0  White             0              40   
#1         10   50           NaN  White             0              13   
#2          8   38           NaN  White             0              40   
#3          6   53           NaN  Black             0              40   
#4         10   28           0.0  Black             0              40   
#
#   classification  sex_0  sex_1  
#0               0    0.0    1.0  
#1               0    0.0    1.0  
#2               0    0.0    1.0  
#3               0    0.0    1.0  
#4               0    1.0    0.0  

