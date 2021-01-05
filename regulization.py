import pickle
import numpy as np
from sklearn import linear_model


pick_train=open('train_pickle','rb')
[x_train,y_train]=pickle.load(pick_train)
pick_test=open('test_pickle','rb')
[x_test,y_test]=pickle.load(pick_test)
n_alphas=50
alphas=np.LogSpace(-5,5,n_alphas)
ridge=linear_model.Ridge()
coefs=[]
erors=[]
p
ick_euror=open('euror_pickle','rb')
baseline_error=pickle.load(pick_euror)
for a in alphas :
    ridge.set_params(alpha=a)
    ridge.fit(x_train,y_train)
    coefs.append(ridge.coef_)
    errors.append([baseline_error, np.mean((ridge.predict(X_test) - y_test) ** 2)]

pick_coefs=open("coefs_pickle","wb")
pickle.dump(coefs,pick_coefs)
#pick_alphas=open("alphas_pickle","wb")
#pickle.dump(alphas,pick_alphas)
#pick_erors=open("alphas_erors","wb")
#pickle.dump(alphas,pick_erors)
#ax=plt.gca()
#ax.plot(alphas,erors,[10**-5,10**5],[baseline_error,baseline_error])
#ax.set_x_scale('log')
#plt.show() 





######ploting results
#draw_first= plt.gca()

#ax.plot(alphas, coefs)
#ax.set_xscale('log')
#plt.xlabel('alpha')
#plt.ylabel('weights')
#plt.title('Ridge coefficients as a function of the regularization')
#plt.axis('tight')
#plt.show()
