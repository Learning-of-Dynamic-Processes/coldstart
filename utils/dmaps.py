import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split

import utils.dm.diffusion_maps as diffusion_maps
import utils.dm.geometric_harmonics as geometric_harmonics


def get_hidden_states(dataset, model):
    """Get the cts corresponding to the xts."""
    hidden_states = (
        model.net.forward(torch.tensor(dataset.input_data, dtype=torch.get_default_dtype()), return_states=True)[0]
        .detach()
        .cpu()
        .numpy()
    )
    hidden_states = np.concatenate((np.zeros_like(hidden_states[:, :1]), hidden_states), axis=1)[:, :-1]
    return hidden_states


def nystrom(Xnew, EigVec, Data, EigenVal, eps):
    epsq = (eps) * 2
    [Nsamp, dsmall] = EigVec.shape
    phi = np.zeros((dsmall, 1))
    dist = cdist(Xnew, Data, metric="sqeuclidean")
    dist = np.array(dist)
    w = np.exp((-dist) / epsq)
    Wtotal = np.sum(w)
    for j in range(0, dsmall):
        phi[j] = (np.sum((w[0, :] / (Wtotal) * EigVec[:, j]))) * (1 / EigenVal[j])
    return phi


def create_chunks(data, max_n_transients, length_chunks, shift_betw_chunks):
    chunks = []
    for i in range(int(min(len(data), max_n_transients))):
        for j in range(int((len(data[i]) - length_chunks) / shift_betw_chunks) + 1):
            chunks.append(data[i][int(j * shift_betw_chunks) : int(j * shift_betw_chunks + length_chunks)])
    chunks = np.squeeze(np.array(chunks))
    return chunks


def dmaps(data, eps=None, return_eps=False):
    """Do diffusion maps on data."""
    pw_dists = pdist(data, "euclidean")
    if eps is None:
        eps = np.median(pw_dists**2)  # scale parameter
    dmap = diffusion_maps.SparseDiffusionMaps(
        points=data,
        epsilon=eps,
        num_eigenpairs=12,
        cut_off=np.inf,
        renormalization=0,
        normalize_kernel=True,
        use_cuda=False,
    )
    V = dmap.eigenvectors.T
    D = dmap.eigenvalues.T
    if return_eps:
        return D, V, eps
    return D, V


def create_geometric_harmonics(dataset_train, dataset_test, config, model):
    c_train = get_hidden_states(dataset_train, model)
    c_test = get_hidden_states(dataset_test, model)

    x_data_train = dataset_train.input_data[:, config["GH"]["initial_set_off"] :]
    c_data_train = c_train[:, config["GH"]["initial_set_off"] :]

    x_data_test = dataset_test.input_data[:, config["GH"]["initial_set_off"] :]
    c_data_test = c_test[:, config["GH"]["initial_set_off"] :]

    x_chunks_train = create_chunks(
        x_data_train,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        config["GH"]["shift_betw_chunks"],
    )
    c_chunks_train = create_chunks(
        c_data_train,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        config["GH"]["shift_betw_chunks"],
    )
    c_chunks_train = c_chunks_train[:, 0, :]

    x_chunks_test = create_chunks(
        x_data_test,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        int(5 * config["GH"]["shift_betw_chunks"]),
    )
    c_chunks_test = create_chunks(
        c_data_test,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        int(5 * config["GH"]["shift_betw_chunks"]),
    )
    c_chunks_test = c_chunks_test[:, 0, :]

    # Diffusion maps on input data
    D, V, eps = dmaps(x_chunks_train, return_eps=True)
    V = V[:, [1, 3]]
    D = D[[1, 3]]

    print("Creating geometric harmonics.")
    V_train, V_test, c_chunks_train_train, c_chunks_train_test = train_test_split(
        V, c_chunks_train, random_state=np.random.seed(10), train_size=8 / 10
    )

    pw = pdist(V_train, "euclidean")
    eps_GH = np.median(pw**2) * 0.05

    GH = diffusion_maps.SparseDiffusionMaps(
        points=V_train,
        epsilon=eps_GH,
        num_eigenpairs=config["GH"]["gh_num_eigenpairs"],
        cut_off=np.inf,
        renormalization=0,
        normalize_kernel=False,
    )
    print("Interpolation function.")
    interp_c = geometric_harmonics.GeometricHarmonicsInterpolator(
        points=V_train, epsilon=None, values=c_chunks_train_train, diffusion_maps=GH
    )
    return V, D, eps, x_chunks_train, c_chunks_train, interp_c


def create_geometric_harmonics_lorenz(dataset_train, dataset_test, config, model):
    c_train = get_hidden_states(dataset_train, model)
    c_test = get_hidden_states(dataset_test, model)

    x_data_train = dataset_train.input_data[:, config["GH"]["initial_set_off"] :]
    c_data_train = c_train[:, config["GH"]["initial_set_off"] :]

    x_data_test = dataset_test.input_data[:, config["GH"]["initial_set_off"] :]
    c_data_test = c_test[:, config["GH"]["initial_set_off"] :]

    x_chunks_train = create_chunks(
        x_data_train,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        config["GH"]["shift_betw_chunks"],
    )
    c_chunks_train = create_chunks(
        c_data_train,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        config["GH"]["shift_betw_chunks"],
    )
    c_chunks_train = c_chunks_train[:, 0, :]

    x_chunks_test = create_chunks(
        x_data_test,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        int(config["GH"]["shift_betw_chunks"]),
    )
    c_chunks_test = create_chunks(
        c_data_test,
        config["GH"]["max_n_transients"],
        config["GH"]["gh_lenght_chunks"],
        int(config["GH"]["shift_betw_chunks"]),
    )
    c_chunks_test = c_chunks_test[:, 0, :]

    # Diffusion maps on input data
    D, V, eps = dmaps(x_chunks_train, return_eps=True)
    # V = V[:, [1, 4, 8]]
    # D = D[[1, 4, 8]]
    V = V[:, 1:9]
    D = D[1:9]

    print("Creating geometric harmonics.")
    V_train, V_test, c_chunks_train_train, c_chunks_train_test = train_test_split(
        V, c_chunks_train, random_state=np.random.seed(10), train_size=8 / 10
    )

    pw = pdist(V_train, "euclidean")
    eps_GH = np.median(pw**2) * 0.05

    GH = diffusion_maps.SparseDiffusionMaps(
        points=V_train,
        epsilon=eps_GH,
        num_eigenpairs=config["GH"]["gh_num_eigenpairs"],
        cut_off=np.inf,
        renormalization=0,
        normalize_kernel=False,
    )
    print("Interpolation function.")
    interp_c = geometric_harmonics.GeometricHarmonicsInterpolator(
        points=V_train, epsilon=None, values=c_chunks_train_train, diffusion_maps=GH
    )
    return V, D, eps, x_chunks_train, c_chunks_train, interp_c
