import logging
import tensorflow as tf

from .utils import chi2_logs
from .utils import monitor_validation

from dataclasses import dataclass


log = logging.getLogger(__name__)

optimizer_options = {
    "Adam": tf.keras.optimizers.Adam,
    "Nadam": tf.keras.optimizers.Nadam,
    "Adadelta": tf.keras.optimizers.Adadelta,
}


@dataclass
class ModelInfo:
    """Class to collect information about the best model found during training"""

    epoch: int = None
    vl_chi2: float = None
    tr_chi2: float = None
    model: tf.keras.Model = None

    def collect_info(self, train_info, vl_chi2, epoch):
        if self.vl_chi2 == None or self.vl_chi2 > vl_chi2:
            self.epoch = epoch
            self.vl_chi2 = vl_chi2
            self.tr_chi2 = train_info.history["loss"][0]
            self.model = train_info.model


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

    kinematics_array = [tf.expand_dims(i, axis=0) for i in kinematics]

    best_model = FitInfo()
    patience_epochs = int(stopping_patience * epochs)
    for epoch in range(epochs):
        train_info = tr_model.fit(
            kinematics_array,
            y=fit_dict["tr_expdat"],
            epochs=1,
            verbose=0,
        )

        vl_chi2 = monitor_validation(
            vl_model, kinematics_array, fit_dict["vl_expdat"]
        )

        best_model.collect_info(train_info, sum(vl_chi2), epoch)

        if not (epoch % 100):
            chi2_logs(train_info, vl_chi2, epoch)

        # If vl chi2 has not improved for a number of epochs equal to
        # `patience_epochs`, stop the fit.
        if epoch - bestmodel.epoch > patience_epochs:
            break

    return best_model
