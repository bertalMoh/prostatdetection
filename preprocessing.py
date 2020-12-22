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
last=input("epreprocessing  finiched , do you want to show x training data ?   ")
if last==yes or last==YES or last==y or last==Y :
    x_train.head()


