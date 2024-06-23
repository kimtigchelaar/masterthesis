import helper
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter('ignore')
 

# create the flower client
class FlowerClient(fl.client.NumPyClient): 

    # get current local model parameters
    def get_parameters(self, config):
        print(f"Client {client_id} received the parameters.")
        return helper.get_params(model)

    # train local model, return model parameters to the server
    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        helper.set_params(model, parameters)
        print("Parameters after setting: ", model.get_params())

        model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}.")

        trained_params = helper.get_params(model)
        print("Trained Parameters: ", trained_params)

        return trained_params, len(X_train), {}

    # evaluate local model, return evaluation result to the server
    def evaluate(self, parameters, config):
        helper.set_params(model, parameters)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_2 = r2_score(y_test, y_pred)

        line = "-" * 21
        print(line)
        print(f"RMSE: {rmse:.8f}")
        print(f"R2  : {r_2:.8f}")
        print(line)

        return mse, len(X_test), {"RMSE": rmse, "R2": r_2}


if __name__ == "__main__":

    # third client 
    client_id = 3
    print(f"Client {client_id}:\n")

    # get dataset for local model
    X_train, y_train, X_test, y_test = helper.load_dataset(client_id - 1)

    # print number of rows
    print("Number of rows in train set:", len(X_train))
    print("Number of rows in test set:", len(X_test))

    # create and fit the local model
    model = RandomForestRegressor(
        criterion='squared_error',
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
    )
    model.fit(X_train, y_train)

    # start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
