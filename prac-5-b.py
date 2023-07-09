#Aim: Evaluating feed forward deep network for multiclass Classification using KFold cross-validation.
#this has alternate version in prac-5-b-alternate.py

#loading libraries
import pandas
from keras.models import Sequential
from keras.layers import Dense
#tensorflow.keras.wrappers.scikit_learn
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

#loading dataset
df=pandas.read_csv('flower.csv',header=0)
print(df)

#splitting dataset into input and output variables
X = df.iloc[:,0:4].astype(float)
y=df.iloc[:,4]
#print(X)
#print(y)

#encoding string output into numeric output
encoder=LabelEncoder()
encoder.fit(y)
encoded_y=encoder.transform(y)
print(encoded_y)
dummy_Y=np_utils.to_categorical(encoded_y)
print(dummy_Y)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator=baseline_model()
estimator.fit(X,dummy_Y,epochs=100,shuffle=True)
action=estimator.predict(X)

for i in range(25):
    print(dummy_Y[i])

print('^^^^^^^^^^^^^^^^^^^^^^')

for i in range(25):
    print(action[i])