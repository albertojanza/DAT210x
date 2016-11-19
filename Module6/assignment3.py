import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

#Load up the /Module6/Datasets/parkinsons.data data set into a variable X, being sure to drop the name column.
#Splice out the status column into a variable y and delete it from X.
df = pd.read_csv("Module6/Datasets/parkinsons.data")
X = df.drop('name',1)
y = X.status
X = X.drop('status',1)

#Perform a train/test split. 30% test group size, with a random_state equal to 7.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

#Create a SVC classifier. Don't specify any parameters, just leave everything as default. 
#Fit it against your training data and then score your testing data

model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test,y_test)
print score
# 0.813559322034





#kernel = "linear"
#kernel = "poly"
kernel = "rbf"
model = SVC(kernel=kernel,C=1,gamma=0.001)


print "Training SVC Classifier..."
#
# .. your code here ..
model.fit(X_train, y_train)




# TODO: Calculate the score of your SVC against the testing data
print "Scoring SVC Classifier..."
score = model.score(X_test,y_test)



