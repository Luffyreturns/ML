import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv(r"C:\Users\omkar\OneDrive\Desktop\ML\dataset\BostonHousing.csv")

Y = data['medv']
X = data.drop('medv', axis=1)

X['rm'] = X['rm'].fillna(X['rm'].mean())
print(X.isnull().sum())

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.3,random_state=42)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_Pred = model.predict(X_test)
 
mse = mean_squared_error(Y_test,Y_Pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(Y_test, Y_Pred)
print(f"R-squared (Accuracy) of the model: {r2:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_test, color='blue', label='Actual Values', alpha=0.6)
plt.scatter(Y_test, Y_Pred, color='red', label='Predicted Values', edgecolor='k', alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('All Parameters - Actual vs Predicted Values')
plt.legend()
plt.show()