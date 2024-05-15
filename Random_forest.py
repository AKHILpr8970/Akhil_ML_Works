import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_set = pd.read_csv("train_and_test2.csv")
data_set.dropna(inplace=True)
df = pd.DataFrame(data_set)
print("Actual dataset")
print(df.to_string())
x = data_set.iloc[:, 1:4].values
y = data_set.iloc[:, 27].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="entropy")

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("----------prediction---------")
df2 = pd.DataFrame({"Actual y test": y_test, "predicted y test": y_pred})
print(df2.to_string())

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
