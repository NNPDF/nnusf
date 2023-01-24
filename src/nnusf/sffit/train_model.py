# -*- coding: utf-8 -*-
"""Compile and train the models."""

import logging

import tensorflow as tf
from rich.live import Live

from .callbacks import (
    EarlyStopping,
    GetTrainingInfo,
    LiveUpdater,
    LogTrainingHistory,
)
from .utils import TrainingStatusInfo, chi2_logs

_logger = logging.getLogger(__name__)


def perform_fit(
    fit_dict,
    data_info,
    replica_dir,
    log_freq,
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
    datas_name = {}
    tr_kinematics = []
    vl_kinematics = []
    for data in data_info.values():
        datas_name[data.name] = 1
        tr_kinematics.append(data.kinematics[data.tr_filter])
        vl_kinematics.append(data.kinematics[~data.tr_filter])

    datas_name["loss"] = 1
    dummy_vl = [1 for _ in range(len(tr_kinematics))]

    # Initialize a placeholder table for `rich` outputs
    table = chi2_logs(datas_name, dummy_vl, datas_name, datas_name, 0)

    tr_kinarray = [tf.expand_dims(i, axis=0) for i in tr_kinematics]
    vl_kinarray = [tf.expand_dims(i, axis=0) for i in vl_kinematics]

    # Initialize callbacks
    train_info_class = TrainingStatusInfo(
        tr_dpts=fit_dict["tr_datpts"], vl_dpts=fit_dict["vl_datpts"]
    )
    stopping = EarlyStopping(
        vl_model,
        stopping_patience,
        val_chi2_threshold,
        train_info_class,
    )
    get_train_info = GetTrainingInfo(
        vl_model, vl_kinarray, fit_dict["vl_expdat"], train_info_class
    )
    log_train_info = LogTrainingHistory(replica_dir, train_info_class, log_freq)

    with Live(table, auto_refresh=False) as rich_live_instance:
        live_updater = LiveUpdater(
            print_rate, train_info_class, table, rich_live_instance
        )

        _logger.info("Start of the training:")
        tr_model.fit(
            tr_kinarray,
            y=fit_dict["tr_expdat"],
            epochs=epochs,
            verbose=0,
            callbacks=[
                get_train_info,
                log_train_info,
                stopping,
                live_updater,
            ],
        )
