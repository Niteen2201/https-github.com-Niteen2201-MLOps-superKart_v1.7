import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load preprocessed splits
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# Identify feature groups correctly
numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
categorical_features = ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]

# Define preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Instantiate a Regressor instead of a Classifier
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Parameter grid for tuning
param_grid = {
    "xgbregressor__n_estimators": [50, 100],
    "xgbregressor__max_depth": [3, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
}

pipeline = make_pipeline(preprocessor, model)

# Optimize for regression (r2 score or neg_mean_squared_error)
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(Xtrain, ytrain)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# Evaluate the model
predictions = best_model.predict(Xtest)
mse = mean_squared_error(ytest, predictions)
r2 = r2_score(ytest, predictions)
print(f"Test Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")
print(f"Test R2 Score: {r2:.4f}")

# Save the best regression model with the filename expected by pipeline.yml
joblib.dump(best_model, "deployment/best_superkart_model_v1.joblib")
print("Model saved to deployment/best_superkart_model_v1.joblib")
