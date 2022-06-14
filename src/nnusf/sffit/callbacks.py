import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, vl_model, patience_epochs, vl_kinematics_array, vl_expdata):
        super().__init__()
        self.vl_model = vl_model
        self.patience_epochs = patience_epochs
        self.vl_kinematics_array = vl_kinematics_array
        self.vl_expdata = vl_expdata
        self.best_epoch = None
        self.best_chi2 = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        chi2 = self.vl_model.evaluate(
            self.vl_kinematics_array, y=self.vl_expdata, verbose=0
        )
        if self.best_chi2 == None or chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        # log.info(f"vl chi2 @ {epoch}: {chi2}")
        epochs_since_best_vl_chi2 = epoch - self.best_epoch
        if epochs_since_best_vl_chi2 > self.patience_epochs:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        log.info(f"Best vl chi2: {self.best_chi2}")
        self.model.set_weights(self.best_weights)
