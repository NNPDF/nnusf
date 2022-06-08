import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, vl_model, patience_epochs, vl_kinematics_array, y):
        super().__init__()
        self.vl_model = vl_model
        self.patience_epochs = patience_epochs
        self.vl_kinematics_array = vl_kinematics_array
        self.best_epoch = None
        self.best_chi2 = None
        self.y = y
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        chi2 = self.vl_model.evaluate(
            self.vl_kinematics_array, self.y, verbose=0
        )
        if self.best_chi2 == None or chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        epochs_since_best_vl_chi2 = epoch - self.best_epoch
        if epochs_since_best_vl_chi2 > self.patience_epochs:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        log.info(f"best epoch: {self.best_epoch}")
        self.model.set_weights(self.best_weights)
