'''
Refer to http://blog.fastforwardlabs.com/2016/02/24/hello-world-in-keras-or-scikit-learn-versus.html
this one is http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
http://cs231n.github.io/
http://cs231n.github.io/neural-networks-1/
'''

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("../data/iris.data", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#print(dummy_y)
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='sigmoid'))
	model.add(Dense(3, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model = baseline_model()
model.fit(X, dummy_y, nb_epoch=200, batch_size=5, verbose=1)

pre = model.predict(X)
print(numpy.argmax(pre, axis=1))

score = model.evaluate(X, dummy_y, verbose=0)
print(score)

