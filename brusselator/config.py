import torch

config = {}
config["DATA"] = {}
config["DATA"]["n_train"] = 600
config["DATA"]["n_val"] = 8
config["DATA"]["n_test"] = 8
config["DATA"]["l_trajectories"] = 150
config["DATA"]["l_trajectories_test"] = 200
config["DATA"]["parameters"] = {}
config["DATA"]["parameters"]["a"] = 1.0
config["DATA"]["parameters"]["b"] = 2.1
config["DATA"]["max_warmup"] = 50
config["PATH"] = "brusselator/models/"

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 4000
config["TRAINING"]["batch_size"] = 256
config["TRAINING"]["learning_rate"] = 5e-3
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["offset"] = 1
config["TRAINING"]["device"] = "cpu"

config["GH"] = {}
config["GH"]["initial_set_off"] = 20
config["GH"]["gh_lenght_chunks"] = 5
config["GH"]["lenght_chunks"] = 10
config["GH"]["max_n_transients"] = 200
config["GH"]["shift_betw_chunks"] = 4
config["GH"]["gh_num_eigenpairs"] = 100

config["MODEL"] = {}
config["MODEL"]["input_size"] = 1
config["MODEL"]["hidden_size"] = []
config["MODEL"]["reservoir_size"] = 2**10
config["MODEL"]["scale_rec"] = 0.9805780023782984
config["MODEL"]["scale_in"] = 0.12843039315361987
config["MODEL"]["leaking_rate"] = 0.5068542695918522
config["TRAINING"]["ridge_factor"] = 1e-2
