import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
tables = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')	
df = tables[0]

# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
# .. your code here ..
df.columns = ['RK','PLAYER','TEAM','GP','G','A','PTS','+/-','PIM','PTS/G','SOG', 'PCT', 'GWG', 'PP_G','PP_A','SH_G','SH_A']


# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..
new_df = df.dropna(thresh=4)

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..
clean_df = new_df[new_df.PLAYER != 'PLAYER']

# TODO: Get rid of the 'RK' column
#
# .. your code here ..
final_df = clean_df.drop('RK',1)

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..
final_df = final_df.reset_index(drop=True)


# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
final_df.dtypes
for i in  range(2,final_df.shape[1]):
  final_df.iloc[:, i] = pd.to_numeric(final_df.iloc[:, i], errors='coerce') 

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

len(final_df.PCT.unique())
final_df.GP[15] + final_df.GP[16]