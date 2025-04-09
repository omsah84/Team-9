import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

dataset = pd.read_csv("HousePricePrediction.csv")


dataset.drop(['Id'], axis=1, inplace=True)


dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())


new_dataset = dataset.dropna()

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(sparse_output=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out(object_cols)

df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model 1: Support Vector Machine (SVM)
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_SVR = model_SVR.predict(X_valid)
svr_mse = mean_squared_error(Y_valid, Y_pred_SVR)
svr_r2 = r2_score(Y_valid, Y_pred_SVR)
print(f'SVM MSE: {svr_mse}')
print(f'SVM R²: {svr_r2}')

# Model 2: Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_RFR = model_RFR.predict(X_valid)
rfr_mse = mean_squared_error(Y_valid, Y_pred_RFR)
rfr_r2 = r2_score(Y_valid, Y_pred_RFR)
print(f'Random Forest Regressor MSE: {rfr_mse}')
print(f'Random Forest Regressor R²: {rfr_r2}')

# Model 3: Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_LR = model_LR.predict(X_valid)
lr_mse = mean_squared_error(Y_valid, Y_pred_LR)
lr_r2 = r2_score(Y_valid, Y_pred_LR)
print(f'Linear Regression MSE: {lr_mse}')
print(f'Linear Regression R²: {lr_r2}')

# Model 4: CatBoost Regressor
cb_model = CatBoostRegressor(verbose=0)
cb_model.fit(X_train, Y_train)
Y_pred_CB = cb_model.predict(X_valid)
cb_mse = mean_squared_error(Y_valid, Y_pred_CB)
cb_r2 = r2_score(Y_valid, Y_pred_CB)
print(f'CatBoost Regressor MSE: {cb_mse}')
print(f'CatBoost Regressor R²: {cb_r2}')
