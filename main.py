import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC    # Sklearn.SVM
from matplotlib import pyplot   # Matplotlib
from sklearn.model_selection import KFold    # Sklearn.Model_selection
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score    # Sklearn.Model_selection
from sklearn.model_selection import cross_val_score    #TEST_TEST
from sklearn.model_selection import train_test_split    # Sklearn.Model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ----------------------------------Clean and collect data-------------------------------------------------
df = pd.read_csv('auto_costs.csv')
df.name = df.name.map(lambda x: x.replace('-', ' '))
df.cylindernumber = df.cylindernumber.map(lambda x: x.strip())
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('four', '4'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('six', '6'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('five', '5'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('three', '3'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('twelve', '12'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('two', '2'))
df.cylindernumber = df.cylindernumber.map(lambda x: x.replace('eight', '8'))





sort_by_wheelbase = df.sort_values(by=['wheelbase'], ascending=True)
sort_by_carlength = df.sort_values(by=['carlength'], ascending=True)
sort_by_carwidth = df.sort_values(by=['carwidth'], ascending=True)
sort_by_carheight = df.sort_values(by=['carheight'], ascending=True)
sort_by_curbweight = df.sort_values(by=['curbweight'], ascending=True)
sort_by_enginesize = df.sort_values(by=['enginesize'], ascending=True)
sort_by_boreratio = df.sort_values(by=['boreratio'], ascending=True)
sort_by_stroke = df.sort_values(by=['stroke'], ascending=True)
sort_by_compressionratio = df.sort_values(by=['compressionratio'], ascending=True)
sort_by_horsepower = df.sort_values(by=['horsepower'], ascending=True)
sort_by_peakrpm = df.sort_values(by=['peakrpm'], ascending=True)
sort_by_citympg = df.sort_values(by=['citympg'], ascending=True)
sort_by_highwaympg = df.sort_values(by=['highwaympg'], ascending=True)
sort_by_cylindernumber = df.sort_values(by=['cylindernumber'], ascending=True)
sort_by_symboling = df.sort_values(by=['symboling'], ascending=True)

# print(sort_by_cylindernumber.cylindernumber)

# print(df[['name','price']])

# ----------------------------------Plot in one plot-------------------------------------
x = range(len(df['name']))
y_1 = sort_by_symboling.symboling
y_2  = sort_by_symboling.price
#
# fig, ax = plt.subplots()
# ax.plot(y_1, y_2)

# -----------------------------------------Plot in two subplots------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('The first plot is cylindernumber and the second price')
ax1.plot(x, y_1)
# plt.title("wheelbase")
ax2.plot(x, y_2)
# plt.show()

# ----------------------Prediction----------------------
reg = linear_model.LinearRegression()
X = df[['carlength', 'carwidth', 'curbweight', 'enginesize', 'citympg', 'highwaympg', 'cylindernumber']]
y = df.price
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

reg.fit(X_train, y_train)
pred_val = reg.predict([[193.7, 73, 3273.8, 122, 23.5, 33.6, 4]])
print(pred_val)

# ----------------------------Actual and predicted values---------------------
y_pred = reg.predict(X_test)
df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

# -----------------------------Accurace of prediction---------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae}')
print(f'Mean squared error: {mse}')
print(f'Root mean squared error: {rmse}')
# print(reg.coef_)

accurancy = r2_score(y_test, y_pred)
print(f"Bizning predict {round(accurancy, 2)*100}% to'g'ri")




