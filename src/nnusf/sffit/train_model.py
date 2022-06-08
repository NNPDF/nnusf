import tensorflow as tf
from callbacks import EarlyStopping
import numpy as np

optimizer_options = {
    "Adam": tf.keras.optimizers.Adam,
    "Nadam": tf.keras.optimizers.Nadam,
    "Adadelta": tf.keras.optimizers.Adadelta,
}

def perform_fit(tr_model, vl_model, data_info, epochs, stopping_patience, optimizer_parameters, **kwargs):
    "Compile the models and do the fit"
    del kwargs

    opt_class = optimizer_options[optimizer_parameters.pop("optimizer")]
    optimizer = opt_class()

    # The model has output nodes corresponding to the chi2 per experiment
    custom_loss = lambda y_true, y_pred : tf.math.reduce_sum(y_pred)

    tr_model.compile(optimizer=optimizer, loss=custom_loss)
    vl_model.compile(loss=custom_loss)

    tr_kinematics = []
    vl_kinematics = []
    for data in data_info.values():
        tr_kinematics.append(data.kinematics[data.tr_filter])
        vl_kinematics.append(data.kinematics[~data.tr_filter])
    tr_kinematics = np.concatenate(tr_kinematics)
    vl_kinematics = np.concatenate(vl_kinematics)

    tr_kinematics_array = tf.expand_dims(tr_kinematics, axis=0)
    vl_kinematics_array = tf.expand_dims(vl_kinematics, axis=0)

    patience_epochs = int(stopping_patience * epochs)
    early_stopping_callback = EarlyStopping(
        vl_model, patience_epochs, vl_kinematics_array
    )

    tr_model.fit(
        tr_kinematics_array,
        y=tf.constant([0]),
        epochs=epochs,
        verbose=2,
        callbacks=[early_stopping_callback],
    )

