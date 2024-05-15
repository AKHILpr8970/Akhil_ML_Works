import pandas as pd
from sklearn.metrics import accuracy_score
# importing datasets
data_set = pd.read_csv("diabetes.csv")
df = pd.DataFrame(data_set)
print(df.to_string())
# Extracting Independent and dependent Variable
x = data_set.iloc[:, :8].values  # selecting age and estimated salary
y = data_set.iloc[:, -1].values  # purchase status
df2 = pd.DataFrame(x)
print(df2.to_string())
# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
print("x_train b4 scaling..")
df3 = pd.DataFrame(x_train)
print(df3.to_string())
# feature Scaling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
print("x_train after scaling...")
df4 = pd.DataFrame(x_train)
print(df4.to_string())
# Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
# p=2: It is equivalent to the standard Euclidean metric.
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)

print("prediction comparison")
ddf = pd.DataFrame({'y_test': y_test, "y_pred": y_pred})
print(ddf.to_string())

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:%.2f' % (accuracy*100))
test = [[10,128, 75,30,164,28.7,0.256,39]]
k_test = st_x.transform(test)
print(k_test)

pred = classifier.predict(k_test)
print(pred)
