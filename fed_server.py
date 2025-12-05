import flwr as fl


def main():
    # Simple FedAvg strategy with 3 rounds
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # use all available clients each round
        fraction_evaluate=1.0, # all clients for evaluation
        min_fit_clients=1,     # minimum number of clients to be sampled for fit
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    print("Starting Flower server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # <-- number of FL rounds
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
