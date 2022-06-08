import tensorflow as tf
from .model_gen import generate_models
from .callbacks import EarlyStopping

def perform_fit(tr_model, vl_model, max_epochs, patience)

    # The model output is the chi2 so we want to minimize the output directly
    custom_loss = lambda y_true, y_pred : y_pred

    tr_model.compile(optimizer="Adam", loss=custom_loss)
    vl_model.compile(optimizer="Adam", loss=custom_loss)


    tr_kinematics_array = tf.expand_dims(tr_kinematics_array, axis=0)
    vl_kinematics_array = tf.expand_dims(vl_kinematics_array, axis=0)

    patience_epochs = int(patience * max_epochs)
    early_stopping_callback = EarlyStopping(
        vl_model, patience_epochs, vl_kinematics_array, y
    )

    tr_model.fit(
        tr_kinematics_array,
        y=tf.constant([0]),
        epochs=max_epochs,
        verbose=2,
        callbacks=[early_stopping_callback],
    )

