import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import make_scorer
import tensorflow as keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Loading Data
file_path = 'path\\ftse100_merged_data.csv'
ftse100_df = pd.read_csv(file_path)
ftse100_df['Date'] = pd.to_datetime(ftse100_df['Date'])

# Preparing Data
X = ftse100_df.drop(['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
y = ftse100_df['Close']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Model Definitions
lin_reg = LinearRegression()
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model = Sequential([Dense(64, activation='relu', input_shape=(X_train.shape[1],)), Dense(64, activation='relu'), Dense(1)])
model.compile(optimizer='adam', loss='mean_squared_error')

# K-Fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# Perform K-Fold CV on the Linear Regression model
lin_reg_scores = cross_val_score(lin_reg, X_imputed, y, cv=kf, scoring=rmse_scorer)
print(f"Linear Regression CV RMSE: {-lin_reg_scores.mean()} ± {lin_reg_scores.std()}")
# Perform K-Fold CV on the Random Forest model
forest_reg_scores = cross_val_score(forest_reg, X_imputed, y, cv=kf, scoring=rmse_scorer)
print(f"Random Forest CV RMSE: {-forest_reg_scores.mean()} ± {forest_reg_scores.std()}")

# --------------------------------- Model Training and Evaluation ----------------------------------------------
# ----------------------- Linear Regression Model -----------------------
# Linear Regression Model RMSE
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lin_reg_mse = mean_squared_error(y_test, y_pred)
lin_reg_rmse = np.sqrt(lin_reg_mse)
# Linear Regression R2
y_pred_lin = lin_reg.predict(X_test)
lin_reg_mse = mean_squared_error(y_test, y_pred_lin)
lin_reg_rmse = np.sqrt(lin_reg_mse)
lin_reg_r2 = r2_score(y_test, y_pred_lin)
# Linear Regression MAE
lin_reg_mae = mean_absolute_error(y_test, y_pred_lin)
# Linear Regression Results
print(f"Linear Regression MAE: {lin_reg_mae}")
print(f"Linear Regression RMSE: {lin_reg_rmse}")
print(f"Linear Regression R-squared: {lin_reg_r2}")

# ----------------------- Random Forest Model -----------------------
# Random Forest Model RMSE
forest_reg.fit(X_train, y_train)
y_pred_forest = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, y_pred_forest)
forest_rmse = np.sqrt(forest_mse)
# Random Forest R2
y_pred_forest = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, y_pred_forest)
forest_rmse = np.sqrt(forest_mse)
forest_r2 = r2_score(y_test, y_pred_forest)
# Random Forest MAE
forest_mae = mean_absolute_error(y_test, y_pred_forest)
# Random Forest Results
print(f"Random Forest MAE: {forest_mae}")
print(f"Random Forest RMSE: {forest_rmse}")
print(f"Random Forest R-squared: {forest_r2}")

# ----------------------- Deep Learning Model -----------------------
# Deep Learning Model rmse
scaler_dl = StandardScaler()
X_train_scaled = scaler_dl.fit_transform(X_train)
X_test_scaled = scaler_dl.transform(X_test)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=6, validation_split=0.1, verbose=0)
y_pred_dl = model.predict(X_test_scaled).flatten()
dl_mse = mean_squared_error(y_test, y_pred_dl)
dl_rmse = np.sqrt(dl_mse)
# Deep Learning R2
y_pred_dl = model.predict(X_test_scaled).flatten()
dl_mse = mean_squared_error(y_test, y_pred_dl)
dl_rmse = np.sqrt(dl_mse)
dl_r2 = r2_score(y_test, y_pred_dl)
# Deep Learning MAE
dl_mae = mean_absolute_error(y_test, y_pred_dl)
# Deep Learning Results
print(f"Deep Learning MAE: {dl_mae}")
print(f"Deep Learning RMSE: {dl_rmse}")
print(f"Deep Learning R-squared: {dl_r2}")


# ----------------------- SVR Model -----------------------
# Setup for SVR with GridSearchCV
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5, 1],
    'svr__kernel': ['linear', 'rbf']
}
# Creating a pipeline that includes scaling and SVR
svr_pipeline = make_pipeline(StandardScaler(), SVR())
# GridSearchCV setup
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
# Predicting with the best model found by GridSearchCV
best_svr_model = grid_search.best_estimator_
y_pred_svr_optimized = best_svr_model.predict(X_test)
# Calculating RMSE for the optimized SVR model
svr_optimized_mse = mean_squared_error(y_test, y_pred_svr_optimized)
svr_optimized_rmse = np.sqrt(svr_optimized_mse)
# Optimized SVR R2
y_pred_svr_optimized = best_svr_model.predict(X_test)
svr_optimized_mse = mean_squared_error(y_test, y_pred_svr_optimized)
svr_optimized_rmse = np.sqrt(svr_optimized_mse)
svr_optimized_r2 = r2_score(y_test, y_pred_svr_optimized)
# Optimized SVR MAE
svr_optimized_mae = mean_absolute_error(y_test, y_pred_svr_optimized)
# Optimized SVR Results
print(f"Optimized SVR MAE: {svr_optimized_mae}")
print(f"Optimized SVR RMSE: {svr_optimized_rmse}")
print(f"Optimized SVR R-squared: {svr_optimized_r2}")


# Define a function for RMSE to use in cross-validation
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Apply K-Fold CV on Linear Regression, Random Forest, and Optimized SVR models
lin_reg_scores = cross_val_score(lin_reg, X_imputed, y, cv=kf, scoring=rmse_scorer)
forest_reg_scores = cross_val_score(forest_reg, X_imputed, y, cv=kf, scoring=rmse_scorer)
svr_optimized_scores = cross_val_score(best_svr_model, X_imputed, y, cv=kf, scoring=rmse_scorer)
# K-Fold CV Results
print(f"Linear Regression CV RMSE: {-lin_reg_scores.mean()} ± {lin_reg_scores.std()}")
print(f"Random Forest CV RMSE: {-forest_reg_scores.mean()} ± {forest_reg_scores.std()}")
print(f"Optimized SVR CV RMSE: {-svr_optimized_scores.mean()} ± {svr_optimized_scores.std()}")

# Sensitivity Analysis (Example with Random Forest)
# Here you can vary the number of estimators to see the effect on RMSE
estimator_range = [50, 100, 200]
for n_estimators in estimator_range:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"RMSE for {n_estimators} estimators: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Stress Testing with Extreme Values (Example with Random Forest)
# Adding noise to test data
X_test_noisy = X_test + np.random.normal(0, 1, X_test.shape)
y_pred_noisy = forest_reg.predict(X_test_noisy)
print(f"Random Forest RMSE on noisy data: {np.sqrt(mean_squared_error(y_test, y_pred_noisy))}")

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# ----------------------- Linear Regression with TimeSeriesSplit -----------------------
lin_reg_time_scores = cross_val_score(lin_reg, X_imputed, y, cv=tscv, scoring=rmse_scorer)
print(f"Linear Regression Time Series CV RMSE: {-lin_reg_time_scores.mean()} ± {lin_reg_time_scores.std()}")

# ----------------------- Random Forest with TimeSeriesSplit -----------------------
rf_time_scores = cross_val_score(forest_reg, X_imputed, y, cv=tscv, scoring=rmse_scorer)
print(f"Random Forest Time Series CV RMSE: {-rf_time_scores.mean()} ± {rf_time_scores.std()}")

# Deep Learning with TimeSeriesSplit
# Function to create and compile a new Keras model
def create_compile_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = create_compile_model(X_train.shape[1]) # Using the function to create the model

# TimeSeriesSplit for Deep Learning model
tscv = TimeSeriesSplit(n_splits=5)
fold = 1
deep_learning_rmse_scores = []

for train_index, test_index in tscv.split(X_imputed):
    print(f"Processing Fold {fold}...")
    X_train_fold, X_test_fold = X_imputed[train_index], X_imputed[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_test_fold_scaled = scaler.transform(X_test_fold)
    model_fold = create_compile_model(X_train_fold_scaled.shape[1]) # Recreating the model for each fold
    model_fold.fit(X_train_fold_scaled, y_train_fold, epochs=100, batch_size=16, verbose=0)
    y_pred_fold = model_fold.predict(X_test_fold_scaled).flatten()
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
    deep_learning_rmse_scores.append(fold_rmse)
    print(f"Fold {fold} Deep Learning RMSE: {fold_rmse}")
    fold += 1

average_rmse = np.mean(deep_learning_rmse_scores)
std_rmse = np.std(deep_learning_rmse_scores)
print(f"Deep Learning Time Series CV RMSE: {average_rmse} ± {std_rmse}")

# ----------------------- SVR with TimeSeriesSplit -----------------------
svr_pipeline = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
svr_time_scores = cross_val_score(svr_pipeline, X_imputed, y, cv=tscv, scoring=rmse_scorer)
print(f"SVR Time Series CV RMSE: {-svr_time_scores.mean()} ± {svr_time_scores.std()}")
