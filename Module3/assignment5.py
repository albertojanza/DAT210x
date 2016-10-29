#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..

df = pd.read_csv('/home/alberto/workspace/DAT210x/Module3/Datasets/wheat.data')


#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here ..
df_drop = df.drop(df.columns[[1, 2, 0]], axis=1) 


#
# TODO: Plot a Andrew's Curve chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
andrews_curves(df_drop, 'wheat_type', alpha=0.4)
plt.show()



#Remember how you dropped the id, area, and perimeter features 
#from your dataset? Well, add back in just the area and perimeter 
#features, re-run your assignment again, then answer the second question below



