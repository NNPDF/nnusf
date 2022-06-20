import logging
import tensorflow as tf

from rich.live import Live

from nnusf.sffit.utils import chi2_logs

from .callbacks import AdaptLearningRate
from .callbacks import EarlyStopping


_logger = logging.getLogger(__name__)


def perform_fit(
    fit_dict,
    data_info,
    epochs,
    stopping_patience,
    optimizer_parameters,
    val_chi2_threshold,
    **kwargs,
):
    "Compile the models and do the fit"
    del kwargs

    opt_name = optimizer_parameters.pop("optimizer", "Adam")
    optimizer = getattr(tf.keras.optimizers, opt_name)
    optimizer = optimizer(**optimizer_parameters)

    tr_model = fit_dict["tr_model"]
    vl_model = fit_dict["vl_model"]

    tr_model.compile(optimizer=optimizer, loss=fit_dict["tr_losses"])
    vl_model.compile(optimizer=optimizer, loss=fit_dict["vl_losses"])
    # tr_model.summary()
    _logger.info("PDF model generated successfully.")

    kinematics = []
    datas_name = {}
    for data in data_info.values():
        kinematics_arr = data.kinematics
        datas_name[data.name] = 1
        kinematics.append(kinematics_arr)
    datas_name["loss"] = 1
    dummy_vl = [1 for _ in range(len(kinematics))]

    lr = optimizer_parameters["learning_rate"]
    table = chi2_logs(datas_name, dummy_vl, datas_name, datas_name, 0, lr)

    kinematics_array = [tf.expand_dims(i, axis=0) for i in kinematics]

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
        )

        _logger.info("Start of the training:")
        train_info = tr_model.fit(
            kinematics_array,
            y=fit_dict["tr_expdat"],
            epochs=epochs,
            verbose=0,
            callbacks=[adapt_lr, stopping],
        )
