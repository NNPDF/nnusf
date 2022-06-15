import logging
import numpy as np
import tensorflow as tf

from nnusf.sffit.callbacks import EarlyStopping
from nnusf.sffit.utils import chi2_logs
from nnusf.sffit.utils import monitor_validation

log = logging.getLogger(__name__)

optimizer_options = {
    "Adam": tf.keras.optimizers.Adam,
    "Nadam": tf.keras.optimizers.Nadam,
    "Adadelta": tf.keras.optimizers.Adadelta,
}


def perform_fit(
    fit_dict,
    data_info,
    epochs,
    stopping_patience,
    optimizer_parameters,
    **kwargs,
):
    "Compile the models and do the fit"
    del kwargs

    optimizer = optimizer_options[
        optimizer_parameters.pop("optimizer")
    ]
    optimizer = optimizer(**optimizer_parameters)

    tr_model = fit_dict["tr_model"]
    vl_model = fit_dict["vl_model"]

    tr_model.compile(optimizer=optimizer, loss=fit_dict["tr_losses"])
    vl_model.compile(optimizer=optimizer, loss=fit_dict["vl_losses"])
    tr_model.summary()

    kinematics = []
    for data in data_info.values():
        kinematics.append(data.kinematics)

    kinematics_array = [tf.expand_dims(i, axis=0) for i in kinematics]

    # TODO: Monitor training & validation losses (callbacks, etc.)
    for epoch in range(epochs):
        train_info = tr_model.fit(
            kinematics_array,
            y=fit_dict["tr_expdat"],
            epochs=1,
            verbose=0,
        )

        if not (epoch % 100):
            # Check validation loss
            vl_chi2 = monitor_validation(
                vl_model, kinematics_array, fit_dict["vl_expdat"]
            )
            chi2_logs(train_info, vl_chi2, epoch)
