#Implementing deep neural network for performing classification task.
'''Problem statement: the given dataset comprises of health information about diabetic women patient.
we need to create deep feed forward network that will classify women suffering from diabetes mellitus as 1.'''

from numpy import loadtxt
from keras.layers import Dense
from keras.models import Sequential

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)

accuracy = model.evaluate(X, Y)

print("accuracy of the model is", (accuracy * 100))
#print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1] * 100))