import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
# .. your code here ..
df = pd.read_csv('/home/alberto/workspace/DAT210x/Module2/Datasets/servo.data', names=['motor', 'screw', 'pgain', 'vgain', 'class'])



# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
# .. your code here ..
vagain_5 = df[(df.vgain == 5)]
len(vagain_5)
# 22

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
# .. your code here ..
len(df[(df.motor == 'E') & (df.screw == 'E')])
# 6

# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
# .. your code here ..
pgain_4 = df[(df.pgain == 4)]
pgain_4.vgain.mean()
# 2.0606060606060606


# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!
pgain_4.dtypes

#motor     object
#screw     object
#pgain      int64
#vgain      int64
#class    float64
#dtype: object

