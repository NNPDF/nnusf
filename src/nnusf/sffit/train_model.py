import logging
import numpy as np
import tensorflow as tf

from nnusf.sffit.callbacks import EarlyStopping
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

    optimizer = optimizer_options[optimizer_parameters.pop("optimizer")]
    optimizer = optimizer(**optimizer_parameters)

    tr_model = fit_dict["tr_model"]
    vl_model = fit_dict["vl_model"]

    tr_model.compile(optimizer=optimizer, loss=fit_dict["tr_losses"])
    vl_model.compile(optimizer=optimizer, loss=fit_dict["vl_losses"])
    tr_model.summary()

    kinematics = []
    for data in data_info.values():
        kinematics.append(data.kinematics)

    # TODO: The following needs to be fixed for multi-dataset fit
    kinematics = np.concatenate(kinematics)
    kinematics_array = tf.expand_dims(kinematics, axis=0)

    for epoch in range(epochs):
        train_info = tr_model.fit(
            kinematics_array,
            y=fit_dict["tr_expdat"],
            epochs=1,
            verbose=0,
        )

        if not (epoch % 100):
            nb_dataset_fit = len(train_info.history)
            nset = 1 if nb_dataset_fit == 1 else (nb_dataset_fit - 1)
            tr_chi2 = train_info.history["loss"][0] / nset
            # Check validation loss
            vl_chi2 = monitor_validation(
                vl_model, kinematics_array, fit_dict["vl_expdat"]
            )
            log.info(f"Epoch {epoch:.4e}: tr={tr_chi2:.4e}; vl={vl_chi2:.4e}")
