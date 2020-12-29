import pandas as pd
from sklearn.model_selection import preprocessing as pre
from sklearn.model_selection import train_test_split as spl
raw_data=pd.read_csv('fiche_cansaire', delimiter='\t')
# x and y chose
x=raw_data.iloc[: ,1:-3]
y=raw_data.iloc[:,-2]
#preprocessing 
#1------------------scaling
std_scale=pre.standardscaler().fit(x)
x_scaled=std_scale.transform(x)
#é------------------spliting
x_train,x_test,y_train,y_test=spl(x_scaled ,y,test_size=0.33)
file_x_train=open("x_train.txt","w")
#save result as pickle file 
import pickle
with open('train.pickle','wb') as f :
    pickle.dump([x_train,y_train],f)
with open('test.pickle','wb') as ts:
    pickle.dump([x_test,y_test],ts)


