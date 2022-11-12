# -*- coding: utf-8 -*-
import time

import numpy as np
import yaml
from hyperopt import STATUS_OK, fmin, hp, space_eval, tpe

from .filetrials import FileTrials
from .model_gen import generate_models
from .train_model import perform_fit


def construct_hyperspace(hyperscan={}, fit_parameters={}, **kwargs):
    del kwargs

    optimizer = hp.choice("optimizer", hyperscan["optimizers"])
    clipnorm = hp.choice("clipnorm", hyperscan["clipnorms"])
    lr = hp.choice("lr", hyperscan["initial_lr"])
    activation = hp.choice("activation", hyperscan["activations"])
    act_out = hp.choice("act_out", hyperscan["activations_output"])

    # Select randomly the number of hidden layers
    np_rng_state = np.random.get_state()
    np.random.seed(seed=int(time.time()))
    nb_hidden = np.random.randint(
        hyperscan["number_layers"]["min"],
        hyperscan["number_layers"]["max"] + 1,
    )
    np.random.set_state(np_rng_state)
    print(f"Number of hidden layers: {nb_hidden}")

    # Add the information ot `hp` for plotting later
    _nbhidden = hp.choice("hidden_layers", [nb_hidden])

    # Select the number of nodes per layer
    nodes_per_layer = []
    for idx, _ in enumerate(range(nb_hidden)):
        nodes_per_layer.append(
            hp.uniformint(
                f"node_{idx}",
                hyperscan["number_nodes"]["min"],
                hyperscan["number_nodes"]["max"],
            )
        )

    # Construct the activation per layer
    actlst = [activation for _ in range(nb_hidden - 1)]
    actlst += [act_out]

    opts = {"optimizer": optimizer, "clipnorm": clipnorm, "learning_rate": lr}

    fit_parameters["units_per_layer"] = nodes_per_layer
    fit_parameters["optimizer_parameters"] = opts
    fit_parameters["activation_per_layer"] = actlst

    return fit_parameters


def perform_hyperscan(hyperfunction, hyperspace, max_evals, folder):
    """
    Parameters
    ----------
    hyperfunction : nnu.train_model.performfit
        function that takes care of the training
    hyperspace :
        hyperspace
    max_evals : int
        total number of evalutions
    cluster : str
        cluster adresses
    folder : str
        folder to store the results
    """

    trials = FileTrials(folder, "trials", parameters=hyperspace)

    best_hyperparameters = fmin(
        fn=hyperfunction,
        space=hyperspace,
        max_evals=max_evals,
        algo=tpe.suggest,
        trials=trials,
    )

    # Save the overall best model into files
    best_setup = space_eval(hyperspace, best_hyperparameters)
    with open("%s/best-model.yaml" % folder, "w") as wfp:
        yaml.dump(best_setup, wfp, default_flow_style=False)


def construct_hyperfunc(data_info, hyperspace_dict, replica_dir, log_freq):
    """
    Parameters
    ----------
    fit_dic: dict
        dictionary containing fit info
    """
    fit_dict = generate_models(data_info, **hyperspace_dict)
    result = perform_fit(
        fit_dict, data_info, replica_dir, log_freq, **hyperspace_dict
    )
    return {"loss": result["best_vl_chi2"], "status": STATUS_OK}
