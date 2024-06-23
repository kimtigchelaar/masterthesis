import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List

# function to load the dataset 
def load_dataset(client_id: int):

    client_id += 1 

    # change "DS01" to "DS04" or "DS08" if needed
    subset = "DS01"
    
    df_train = pd.read_csv(f"data/{subset}/client{client_id}_train_{subset}.csv")
    df_test = pd.read_csv(f"data/{subset}/client{client_id}_test_{subset}.csv")

    X_train = df_train.drop(columns=["RUL"])
    X_test = df_test.drop(columns=["RUL"])

    y_train = df_train["RUL"]
    y_test = df_test["RUL"]

    return X_train, y_train, X_test, y_test
    

# get parameters from the RandomForestRegressor
def get_params(model: RandomForestRegressor) -> List[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params


# set the parameters in the RandomForestRegressor
def set_params(model: RandomForestRegressor, params: List[np.ndarray]) -> RandomForestRegressor:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model
