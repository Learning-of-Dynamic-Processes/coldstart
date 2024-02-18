import torch

config = {}
config["DATA"] = {}
config["DATA"]["n_train"] = 600
config["DATA"]["n_val"] = 4
config["DATA"]["n_test"] = 4
config["DATA"]["l_trajectories"] = 100
config["DATA"]["l_trajectories_test"] = 250
config["DATA"]["parameters"] = {}
config["DATA"]["parameters"]["sigma"] = 10.0
config["DATA"]["parameters"]["rho"] = 28.0
config["DATA"]["parameters"]["beta"] = 8.0 / 3.0
config["DATA"]["max_warmup"] = 50
config["DATA"]["load_data"] = False
config["PATH"] = "lorenz/models/"

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 1000
config["TRAINING"]["batch_size"] = 400
config["TRAINING"]["learning_rate"] = 5e-3
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100
config["TRAINING"]["offset"] = 20
config["TRAINING"]["device"] = "cpu"

config["GH"] = {}
config["GH"]["initial_set_off"] = 20
config["GH"]["gh_lenght_chunks"] = 7
config["GH"]["lenght_chunks"] = 10
config["GH"]["max_n_transients"] = 200
config["GH"]["shift_betw_chunks"] = 3
config["GH"]["gh_num_eigenpairs"] = 100

config["MODEL"] = {}
# Number of variables to use when using the lorenz system
config["MODEL"]["input_size"] = 1
config["MODEL"]["hidden_size"] = []
config["MODEL"]["reservoir_size"] = 2**11
config["MODEL"]["scale_rec"] = 0.9
config["MODEL"]["scale_in"] = 0.02
config["MODEL"]["leaking_rate"] = 0.5
config["TRAINING"]["ridge_factor"] = 1e-2

# Train loss: 0.000068
# Val loss: 0.000051
config["MODEL"]["leaking_rate"] = 0.5012724887553683
config["TRAINING"]["ridge_factor"] = 1e-3
config["MODEL"]["scale_rec"] = 0.8048300803863692
config["MODEL"]["scale_in"] = 0.17928881689644652
