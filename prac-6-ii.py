#continuation of prac-6
'''The above code and resultant graph demonstrate overfitting with accuracy of testing data less than accuracy of training data
also the accuracy of testing data increases once and then start decreases gradually.
to solve this problem we can use regularization
Hence, we will add two lines in the above code as highlighted below to implement l2 regularization with alpha=0.001
'''

from matplotlib import pyplot
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
#added line
from keras.regularizers import l2

X,Y=make_moons(n_samples=100,noise=0.2,random_state=1)
n_train=30
trainX,testX=X[:n_train,:],X[n_train:]
trainY,testY=Y[:n_train],Y[n_train:]
#print(trainX)
#print(trainY)
#print(testX)
#print(testY)

model=Sequential()

#replaced line
#model.add(Dense(500,input_dim=2,activation='relu'))
model.add(Dense(500,input_dim=2,activation='relu',kernel_regularizer=l2(0.001)))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=400)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test')
pyplot.legend()
pyplot.show()