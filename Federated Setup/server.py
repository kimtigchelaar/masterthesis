import flwr as fl
from typing import Dict 
import helper
from sklearn.ensemble import RandomForestRegressor


# define the global value for the number of clients and the training round
NUM_CLIENTS = 3

# ranges from 2 to 5 
ROUNDS = 2
   

# return the current round
def fit_config(server_round: int) -> Dict:
    config = {
        "server_round": server_round,
    }
    return config


# aggregate metrics and calculate weighted averages
def metrics_aggregate(results) -> Dict:
    if not results:
        return {}

    else:
        # number of samples in the dataset
        total_samples = 0  

        # collecting metrics
        aggregated_metrics = {
            "RMSE" : 0,
            "R2" : 0
        }

        # get resulting values 
        for samples, metrics in results:
            for key, value in metrics.items():
                if key not in aggregated_metrics:
                    aggregated_metrics[key] = 0
                else:
                    aggregated_metrics[key] += (value * samples)
            total_samples += samples

        # compute weighted average for each metric
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] = round(aggregated_metrics[key] / total_samples, 6)

        return aggregated_metrics



if __name__ == "__main__":

    print(f"Server:\n")

    # determine initial model parameters
    model = RandomForestRegressor(
        criterion='squared_error',
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
    )

    # get model weights as a list of NumPy ndarray's
    weights = helper.get_params(model)

    # serialize ndarrays to `Parameters`
    parameters = fl.common.ndarrays_to_parameters(weights)

    # aggregation strategy (change "FedAvg" to "FedTrimmedAvg", "FedOpt" or "FedYogi")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=metrics_aggregate,
        fit_metrics_aggregation_fn=metrics_aggregate,
        initial_parameters=parameters
    )

    # saving the server log
    fl.common.logger.configure(identifier="FL_Test", filename="log.txt")

    # start server
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        server_address="127.0.0.1:8080",
    )
