import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# Import the dataset
df = pd.read_csv('cali_housing.csv')

# Assign the target (y) and features (X)
y = df['MedHouseVal']
X = df.drop(columns='MedHouseVal')

# Train Test Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# All of the values are numeric, and we are not scaling for this model so we can proceed with the developing the model
reg = LinearRegression()

# Train the model
reg.fit(X_train,y_train)

# Measuring Model Performance
# The R^2 value on our training set was 0.609 and the R^2 value on our test set was 0.591.
# In this case, our train and test scores were similar which means that our model was not "overfit".
train_score = reg.score(X_train, y_train) # ~ 0.609873031052925
test_score = reg.score(X_test,y_test) # ~ 0.5910509795491352

# Obtaining the predictions
train_preds = reg.predict(X_train)
test_preds = reg.predict(X_test)

# Regression Metrics

# Coefficient of Determination (R^2):
r2_train = r2_score(y_train, train_preds)
r2_test = r2_score(y_test, test_preds)

# MAE(Mean Absolute Value) mean of the absolute value of the errors.
mae_train = mean_absolute_error(y_train, train_preds)
mae_test = mean_absolute_error(y_test, test_preds)

# MSE mean of the squared errors. MSE "punishes" larger errors, which tends to be useful in the real world.
mse_train = mean_squared_error(y_train, train_preds)
mse_test = mean_squared_error(y_test, test_preds)

# RMSE (Root mean squared error)
rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))
rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))

