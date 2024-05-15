import numpy as np
import pandas as pd
from sklearn import metrics

data_set=pd.read_csv("diabetes.csv",skiprows=1)
df=pd.DataFrame(data_set)
print(df.to_string)

x=data_set.iloc[:,0:8].values
y=data_set.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
print(log_reg.fit(X_train, y_train))

from  sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred = log_reg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(accuracy_score(y_train, log_reg.predict(X_train)))