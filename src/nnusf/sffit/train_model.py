# -*- coding: utf-8 -*-
"""Compile and train the models."""

import logging

import tensorflow as tf
from rich.live import Live

from .callbacks import AdaptLearningRate, EarlyStopping
from .utils import chi2_logs

_logger = logging.getLogger(__name__)


def perform_fit(
    fit_dict,
    data_info,
    epochs,
    stopping_patience,
    optimizer_parameters,
    val_chi2_threshold,
    print_rate=100,
    **kwargs,
):
    """Compile the models and do the fit."""
    del kwargs

    opt_name = optimizer_parameters.pop("optimizer", "Adam")
    optimizer = getattr(tf.keras.optimizers, opt_name)
    optimizer = optimizer(**optimizer_parameters)

    tr_model = fit_dict["tr_model"]
    vl_model = fit_dict["vl_model"]

    tr_model.compile(optimizer=optimizer, loss=fit_dict["tr_losses"])
    vl_model.compile(optimizer=optimizer, loss=fit_dict["vl_losses"])
    _logger.info("PDF model generated successfully.")

    # Prepare some placeholder values to initialize
    # the printing of `rich` tables.
    kinematics = []
    datas_name = {}
    for data in data_info.values():
        kinematics_arr = data.kinematics
        datas_name[data.name] = 1
        kinematics.append(kinematics_arr)
    datas_name["loss"] = 1
    dummy_vl = [1 for _ in range(len(kinematics))]

    # Initialize a placeholder table for `rich` outputs
    lr = optimizer_parameters["learning_rate"]
    table = chi2_logs(datas_name, dummy_vl, datas_name, datas_name, 0, lr)

    # prepare the inputs, including an input with all x=1 used to enforce F_i(x=1)=0
    kinematics_array = []
    for kinematic_arr in kinematics:
        kinematics_array.append(tf.expand_dims(kinematic_arr, axis=0))

    with Live(table, auto_refresh=False) as rich_live_instance:
        # Instantiate the various callbacks
        adapt_lr = AdaptLearningRate(fit_dict["tr_datpts"])
        stopping = EarlyStopping(
            vl_model,
            kinematics_array,
            fit_dict["vl_expdat"],
            fit_dict["tr_datpts"],
            fit_dict["vl_datpts"],
            stopping_patience,
            val_chi2_threshold,
            table,
            rich_live_instance,
            print_rate,
        )

        _logger.info("Start of the training:")
        tr_model.fit(
            kinematics_array,
            y=fit_dict["tr_expdat"],
            epochs=epochs,
            verbose=0,
            callbacks=[adapt_lr, stopping],
        )

    # Save various metadata into a dictionary
    final_results = {
        "best_tr_chi2": adapt_lr.loss_value,
        "best_vl_chi2": stopping.best_chi2 / stopping.tot_vl,
        "best_epochs": stopping.best_epoch,
    }
    return final_results
