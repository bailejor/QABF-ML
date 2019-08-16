import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy.random import seed
from sklearn.metrics import confusion_matrix

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


#dataframe = pandas.read_csv("Realtest.csv", header=None)
#dataset = dataframe.values
# split into input (X) and output (Y) variables
#X_test = dataset[:,0:5].astype(float)
#y_test = dataset[:,5:10]


#split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state = seed)

model = Sequential()
model.add(Dense(5000, activation='relu', input_dim=X_train.shape[1]))
#model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))



#model.add(Dropout(0.1))

model.add(Dense(y_train.shape[1], activation='sigmoid'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=2000)

preds = model.predict(X_test)
print(preds)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
#score = compare preds and y_test
print(preds)
print(y_test)
c = (np.array(preds) == np.array(y_test))
added = 0
print(c)
for i in c:
	if False in i:
		added = added + 1
print("accuracy is ", (len(y_test)-added)/len(y_test))

interval = 1.77 * sqrt( (0.64 * (1 - 0.64)) / 14)
print('%.3f' % interval)
print(X_test)


