import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

# Look pretty...
matplotlib.style.use('ggplot')


scaleFeatures = True


# TODO: Load up the dataset and remove any and all
# Rows that have a nan. You should be a pro at this
# by now ;-)
#
# .. your code here ..
original_df = pd.read_csv('Datasets/kidney_disease.csv')
original_df.shape
df_without_na = original_df.dropna()
df_without_na.shape

# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in df_without_na.classification]

# TODO: Instead of using an indexer to select just the bgr, rc, and wc, 
#alter your assignment code to drop all the nominal features listed above. 
#Be sure you select the right axis for columns and not rows, otherwise Pandas will complain!

# id,age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane,classification
# ['id', 'classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
df = df_without_na.drop(['id', 'classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], 1)

#Right after you print out your dataset's dtypes, add an exit() so you can inspect the results. 
#Does everything look like it should / properly numeric? If not, make code changes to coerce the 
#remaining column(s).
print df.dtypes
df.wc = df.wc.astype(int)
df.pcv = df.pcv.astype(int)
df.rc = df.rc.astype(float)
print df.dtypes

# TODO: PCA Operates based on variance. The variable with the greatest
# variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results. Also print out the results of running .describe
# on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then
# you probably didn't complete the previous step properly.
#
# .. your code here ..
print df.describe()
print "Variance of wc is %s. " % df.wc.var()
print "Std of wc is %s. " % df.wc.std()
print "Variance of rc is %s. " % df.rc.var()
print "Std of rc is %s. " % df.rc.std()
print "Variance of bgr is %s. " % df.bgr.var()
print "Std of bgr is %s. " % df.bgr.std()



# TODO: This method assumes your dataframe is called df. If it isn't,
# make the appropriate changes. Don't alter the code in scaleFeatures()
# just yet though!
#
# .. your code adjustment here ..

if scaleFeatures: df = helper.scaleFeatures(df)



# TODO: Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
T = pca.fit_transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


#Alter your code so that you only drop the id and classification columns. For the remaining 10 nominal 
#features, properly encode them by as explained in the Feature Representation section by creating new, 
#boolean columns using Pandas .get_dummies(). You should be able to carry that out with a single line of code. 
#Run your assignment again and see if your results have changed at all.
df = df_without_na.drop(['id', 'classification'],axis=1)

print df.dtypes
df.wc = df.wc.astype(int)
df.pcv = df.pcv.astype(int)
df.rc = df.rc.astype(float)
for column in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
  df[column] = df[column].astype("category").cat.codes
print df.dtypes

df = pd.get_dummies(df,columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])


if scaleFeatures: df = helper.scaleFeatures(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
T = pca.fit_transform(df)
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()
