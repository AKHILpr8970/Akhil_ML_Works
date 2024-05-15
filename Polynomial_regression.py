import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
data_set = pd.read_csv("Ice_cream selling data.csv")
data_set.dropna(inplace=True)
df = pd.DataFrame(data_set)
print("Actual dataset")
print(df.to_string())
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_regs = LinearRegression()
lin_regs.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_regs = PolynomialFeatures(degree=2)
x_poly = poly_regs.fit_transform(x)
print(x_poly)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

mtp.scatter(x, y, color="blue")
mtp.plot(x, lin_regs.predict(x), color="red")
mtp.title("Bluff detection model(Linear regression)")
mtp.xlabel("Position levels")
mtp.ylabel("salary")
mtp.show()

mtp.scatter(x, y, color="blue")
mtp.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)),color="red")
mtp.title("Bluff detection model(polynomial Regression)")
mtp.xlabel("Temperature (Â°C)")
mtp.ylabel("Ice Cream Sales (units)")
mtp.show()

lin_pred = lin_regs.predict([[-7.34]])
print(lin_pred)

poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[-7.34]]))
print(poly_pred)
