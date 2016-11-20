import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold


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

#Instead, lets get the computer to do what computers do best. Program a naive, best-parameter search by 
#creating nested for-loops. The outer for-loop should iterate a variable C from 0.05 to 2, using 0.05 unit 
#increments. The inner for-loop should increment a variable gamma from 0.001 to 0.1, using 0.001 unit increments. 
#As you know, Python ranges won't allow for float intervals, so you'll have to do some research on NumPy ARanges, 
#if you don't already know how to use them.



def best_score_svc(kernel,X_train,y_train,X_test,y_test):
  best_score = 0
  for C in numpy.arange(0.05,2,0.05):
    for gamma in numpy.arange(0.001,0.1,0.001):
      model = SVC(kernel=kernel,C=C,gamma=gamma)
      model.fit(X_train, y_train)
      score = model.score(X_test,y_test)
      best_score = score if score > best_score else best_score
  return best_score

#kernel = "poly"
#kernel = "rbf" #0.9152542372881356
#kernel = "linear" # 0.86440677966101698
print best_score_svc("rbf",X_train,y_train,X_test,y_test)

#Right after you preform the train/test split but before you train your model, inject SciKit-Learn's pre-processing 
#code. Unless you have a good idea which one is going to work best, you're going to have to try the various 
#pre-processors one at a time, checking to see if they improve your predictive accuracy.
#scaling = preprocessing.Normalizer() # 0.796610169492
#scaling = preprocessing.MaxAbsScaler() # 0. 881355932203
#scaling = preprocessing.MinMaxScaler() # 0.881355932203
#scaling = preprocessing.KernelCenterer() # 0.915254237288
scaling = preprocessing.StandardScaler() #0.932203389831

scaling.fit(X_train)
T_X_test = scaling.transform(X_test)
T_X_train = scaling.transform(X_train)
print best_score_svc("rbf",T_X_train,y_train,T_X_test,y_test)


#Well, let's try to get rid of some useless features. Immediately after you do the pre-processing, run PCA on your 
#dataset. The original dataset has 22 columns and 1 label column. So try experimenting with PCA n_component values 
#between 4 and 14. Are you able to get a better accuracy?

best_PCA_score = 0
best_PCA_dimension = 0
for dimension in range(4,14):  
  model = PCA(n_components=dimension)
  model.fit(T_X_train)
  PCA_T_X_train = model.transform(T_X_train)
  PCA_T_X_test = model.transform(T_X_test)
  score = best_score_svc("rbf",PCA_T_X_train,y_train,PCA_T_X_test,y_test)
  if score > best_PCA_score :
    best_PCA_score = score
    best_PCA_dimension = dimension
print best_PCA_score # 0.932203389831 with StandardScaler, 
print dimension  # 13

#In the same spot, run Isomap on the data. Manually experiment with every inclusive combination of n_neighbors 
#between 2 and 5, and n_components between 4 and 6. Are you able to get a better accuracy?
best_ISO_score = 0
best_ISO_n_neighbor = 0
best_ISO_n_component = 0

for n_neighbor in range(2,5):
  for n_component in range(4,6):  
    model = manifold.Isomap(n_neighbors=n_neighbor, n_components=n_component)
    model.fit(T_X_train)
    ISO_T_X_train = model.transform(T_X_train)
    ISO_T_X_test = model.transform(T_X_test)
    score = best_score_svc("rbf",ISO_T_X_train,y_train,ISO_T_X_test,y_test) 
    if score > best_ISO_score :
      best_ISO_score = score
      best_ISO_n_component = n_component
      best_ISO_n_neighbor = n_neighbor
print best_ISO_score #0.949152542373 with StandardScaler, 
print best_ISO_n_component # 5
print best_ISO_n_neighbor  #2

