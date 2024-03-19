# Data-driven cold starting of good reservoirs 

INSTALLATION
---------


Clone repository and install dependencies

    git clone https://github.com/Learning-of-Dynamic-Processes/coldstart.git
    cd lpde
    pip install -r requirements.txt

In addition, a running version of pycuda is necessary for the data generation.


USAGE
---------

This python package contains functions to initialize reservoir computers for two examples, the Brusselator and the Lorenz system.

- Run `brusselator_optuna.py` to find a good set of reservoir hyperparamerters for the Brusselator system. Paste the best set of parameters into `brusselator/config.py`.
- Run `brusselator_train.py` to create a reservoir using the parameters in `brusselator/config.py`.
- Use `brusselator.ipynb` to create the cold start map and the Brusselator figures shown in the paper.

Likewise, for the Lorenz system:

- Run `lorenz_optuna.py` to find a good set of reservoir hyperparamerters for the Lorenz system. Paste the best set of parameters into `lorenz/config.py`.
- Run `lorenz_train.py` to create a reservoir using the parameters in `lorenz/config.py`.
- Use `lorenz.ipynb` to create the cold start map and the Lorenz figures shown in the paper.

To recreate results from section 4.1 of the article, use the code in `coldstart_fig_9/`.

LICENCE
---------

This work is licenced under MIT License.
Please cite

"Data-driven cold starting of good reservoirs"
L. Grigoryeva et al., (2023).
*ArXiv* arXiv:2403.10325 
(https://arxiv.org/abs/2403.10325)

if you use this code for publications.

In addition, if you use the diffusion maps or geometric harmonics functions, cite

" datafold: data-driven models for point clouds and time series on manifolds"
Lehmberg et al., (2020).
Journal of Open Source Software, 5(51), 2283, 
(https://doi.org/10.21105/joss.02283)
