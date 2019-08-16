import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy.random import seed


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline


'''from numpy.random import seed
seed(6)
from tensorflow import set_random_seed
set_random_seed(6)      Varied predictions'''

seed = 6
np.random.seed(seed)

dataframe = pandas.read_csv("RealdataIntensity.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_train = dataset[:,0:10].astype(float)
y_train = dataset[:,10:15].astype(float)

def baseline_model():
	model = Sequential()
	model.add(Dense(5000, activation='relu', input_dim=X_train.shape[1]))
	model.add(Dense(500, activation='relu'))
	model.add(Dense(y_train.shape[1], activation='sigmoid'))
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model
	
estimator = KerasClassifier(build_fn = baseline_model, epochs=50, batch_size=2000, verbose = 0)
#kfold = KFold(n_splits=3, shuffle=True, random_state=seed)

kfold = LeaveOneOut()   
'''for leave one out cross fold validation'''
results = cross_val_score(estimator, X_train, y_train, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
