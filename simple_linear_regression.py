import pickle
from sklearn import linear_model
pick_train=open('train_pickle','rb')
[x_train,y_train]=pickle.load(pick_train)
pick_test=open('test_pickle','rb')
[x_test,y_test]=pickle.load(pick_test)
Lr=linear_model.LinearRegression()
Lr.fit(x_train,y_train)
import numpy as np
euror=np.mean((Lr.predict(x_test)-y_test)**2)
print(euror)