from sklearn import linear_model
import pandas as pd
import numpy as np

data = pd.read_csv('../input/train.csv')
data2 = pd.read_csv('../input/test.csv')

data = np.array(data)

y = data[:,1]
pclass = data[:,2]
sex = data[:,4]
age = data[:,5]
sibsp = data[:,6]
parch = data[:,7]
fare = data[:,9]

age_new = pd.DataFrame(age).fillna(method='pad')
age_new = np.mat(age_new)

for i in range(np.size(sex)):
	if(sex[i] == 'male'):
		sex[i] = 1
	else:
		sex[i]= 0

X = [list(pclass), list(sex), list(sibsp), list(parch), list(age_new)]

regr = linear_model.LinearRegression()
regr.fit(np.array(X).transpose(), y)

data2 = np.array(data2)

p_id = data2[:,0]
pclass = data2[:,1]
sex = data2[:,3]
age = data2[:,4]
sibsp = data2[:,5]
parch = data2[:,6]
fare = data2[:,8]

age_new = pd.DataFrame(age).fillna(method='pad')
age_new = np.mat(age_new)

for i in range(np.size(sex)):
	if(sex[i] == 'male'):
		sex[i] = 1
	else:
		sex[i]= 0

X_new = [list(pclass), list(sex), list(sibsp), list(parch), list(age_new)]

Y_new = regr.predict(np.array(X_new).transpose())

a = np.array(p_id)
b = np.array(np.round(Y_new))
p = [a,b]
pd.DataFrame(p).transpose().to_csv("my_solution.csv", index = 0)