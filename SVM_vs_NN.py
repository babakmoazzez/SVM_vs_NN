from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
import numpy as np
import time


#load data library
lib = fetch_lfw_people(resize=1.0,min_faces_per_person=50)

keys = lib.keys()
print(keys)
#dict_keys(['data', 'images', 'target', 'target_names', 'DESCR'])

#prepare data
X = lib['data']
y = lib['target']
names = lib['target_names']

#get sizes
_,width,height = lib['images'].shape
n = X.shape[1] #number of features
m = X.shape[0] #number of training examples
K = lib['target_names'].shape[0] #number of classes

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,test_size = 0.2, shuffle = True)

#Classification using Support Vector Machine 
#Use 5-fold cross validation to find the best regularization parameter and kernel
parameters = {'kernel': ['linear','rbf','sigmoid','poly'], 'C':[0.001,0.01,0.1,1,10,100]}
clf = GridSearchCV(SVC(probability = True),parameters,cv = 5)
SVM = clf.fit(X_train,y_train)
sorted(clf.cv_results_.keys())


print(SVM.best_estimator_)
y_predicted = SVM.predict(X_test)
print(classification_report(y_test,y_predicted,target_names=names))

#ROC curve to be added here

#Classification using Support Vector Machine 
NN={}
score={}
NN['relu']=MLPClassifier(hidden_layer_sizes=(64,128,256,512,1024,1024),activation='relu',solver='lbfgs',max_iter = 1000)
tic = time.time()
NN['relu'].fit(X_train,y_train)
toc = time.time()
score['relu'] = NN['relu'].score(X_test,y_test)
print("score with relu",score['relu']," time: ", 1000000*(toc - tic))

NN['tanh']=MLPClassifier(hidden_layer_sizes=(64,128,256,512,1024,1024),activation='tanh',solver='lbfgs',max_iter = 1000)
tic = time.time()
NN['tanh'].fit(X_train,y_train)
toc = time.time()
score['tanh'] = NN['tanh'].score(X_test,y_test)
print("score with tanh",score['tanh']," time: ", 1000000*(toc - tic))

NN['logistic']=MLPClassifier(hidden_layer_sizes=(64,128,256,512,1024,1024),activation='logistic',solver='lbfgs',max_iter = 1000)
tic = time.time()
NN['logistic'].fit(X_train,y_train)
toc = time.time()
score['logistic'] = NN['logistic'].score(X_test,y_test)
print("score with logistic",score['logistic']," time: ", 1000000*(toc - tic))

print("best NN is with ", max(score,key=score.get))
y_predicted = NN[max(score)].predict(X_test)
print(classification_report(y_test,y_predicted,target_names=names))


#CNNs to be added here soon