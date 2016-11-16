import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 

samples = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
import os
for image in os.listdir('Module4/Datasets/ALOI/32/'):
  samples.append(misc.imread('Module4/Datasets/ALOI/32/' + image).reshape(-1))

df = pd.DataFrame(samples)
from sklearn import manifold

for n_neighbor in range (1,7):
  iso = manifold.Isomap(n_neighbors=n_neighbor, n_components=3)
  iso.fit(df)
  T = iso.transform(df)
  
  # 2D Scatterplot
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title("ISO map on ALOI with {n} neighbors".format(n=n_neighbor))
  ax.set_xlabel('Manifold Component: 0')
  ax.set_ylabel('Manifold Component: 1')
  ax.scatter(T[:,0],T[:,1], marker='.',alpha=0.7)

  # 3D scatterplot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title("ISOMAP 3D map on ALOI with {n} neighbors".format(n=n_neighbor))
  ax.set_xlabel('Manifold Component: 0')
  ax.set_ylabel('Manifold Component: 1')
  ax.set_zlabel('Manifold Component: 2')
  ax.scatter(T[:,0], T[:,1], T[:,2], c='green', marker='.', alpha=0.75)


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
mixed_samples = list(samples)
colors = []
for i in mixed_samples:
  colors.append('b') 
for image in os.listdir('Module4/Datasets/ALOI/32i/'):
  mixed_samples.append(misc.imread('Module4/Datasets/ALOI/32i/' + image).reshape(-1))
  colors.append('r')

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
mixed_df = pd.DataFrame(mixed_samples)


#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
mixed_iso = manifold.Isomap(n_neighbors=6, n_components=3)
mixed_iso.fit(mixed_df)
mixed_T = mixed_iso.transform(mixed_df)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("ISO map on ALOI with 6 neighbors")
ax.set_xlabel('Manifold Component: 0')
ax.set_ylabel('Manifold Component: 1')
ax.scatter(mixed_T[:,0],mixed_T[:,1], marker='.',alpha=0.7,c=colors)


#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 

# 3D scatterplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("ISOMAP 3D map on ALOI with 6 neighbors")
ax.set_xlabel('Manifold Component: 0')
ax.set_ylabel('Manifold Component: 1')
ax.set_zlabel('Manifold Component: 2')
ax.scatter(mixed_T[:,0], mixed_T[:,1], mixed_T[:,2], marker='.', alpha=0.75,c=colors)

