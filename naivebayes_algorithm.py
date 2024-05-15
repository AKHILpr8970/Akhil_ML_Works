import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

data_set = pd.read_csv("train_and_test2.csv")
data_set.dropna(inplace=True)
df = pd.DataFrame(data_set)
print(df.to_string())

x = data_set.iloc[:,1:4].values
y = data_set.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("-------------prediction-------------")
df2 =pd.DataFrame({"Actual": y_test, "predicted": y_pred})
print(df2.to_string())
from sklearn import metrics
print("mean absolute error:", metrics.mean_absolute_error(y_test,y_pred))
print("mean squared error:", metrics.mean_squared_error(y_test, y_pred))
print("root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:%.2f' % (accuracy*100))
