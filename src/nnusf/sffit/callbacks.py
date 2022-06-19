import logging

import tensorflow as tf

from rich.live import Live

from .utils import chi2_logs

_logger = logging.getLogger(__name__)


ADAPTIVE_LR = [
    {"range": [100, 250], "lr": 0.025},
    {"range": [50, 100], "lr": 0.01},
    {"range": [40, 50], "lr": 0.0075},
    {"range": [40, 50], "lr": 0.005},
    {"range": [10, 30], "lr": 0.0025},
    {"range": [5, 10], "lr": 0.0015},
    {"range": [1, 5], "lr": 0.001},
]


def modify_lr(tr_loss_val, lr):
    for dic in ADAPTIVE_LR:
        range, lrval = dic["range"], dic["lr"]
        check = range[0] <= tr_loss_val < range[1]
        if check and (lr > lrval):
            return lrval
    return lr


class AdaptLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, tr_dapts):
        super(AdaptLearningRate, self).__init__()
        self.loss_value = 1e5
        self.nbdpts = sum(tr_dapts.values())

    def on_batch_end(self, batch, logs={}):
        """Update value of LR after each epochs"""
        self.loss_value = logs.get("loss") / self.nbdpts

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer does not have LR attribute.")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = modify_lr(self.loss_value, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        vl_model,
        kinematics,
        vl_expdata,
        tr_dpts,
        vl_dpts,
        patience_epochs,
        table,
        live,
    ):
        super().__init__()
        self.vl_model = vl_model
        self.live = live
        self.table = table
        self.kinematics = kinematics
        self.vl_expdata = vl_expdata
        self.best_epoch = None
        self.best_chi2 = None
        self.best_weights = None
        self.tr_dpts = tr_dpts
        self.vl_dpts = vl_dpts
        self.patience_epochs = patience_epochs

    def on_epoch_end(self, epoch, logs={}):
        chix = self.vl_model.evaluate(self.kinematics, y=self.vl_expdata, verbose=0)
        chi2 = chix[0] if isinstance(chix, list) else chix
        if self.best_chi2 == None or chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

        if not (epoch % 100):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            self.table = chi2_logs(logs, chix, self.tr_dpts, self.vl_dpts, epoch, lr)
            self.live.update(self.table)

        epochs_since_best_vl_chi2 = epoch - self.best_epoch
        check_val = epochs_since_best_vl_chi2 > self.patience_epochs
        if check_val and (self.best_chi2 <= 3):
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        _logger.info(f"best epoch: {self.best_epoch}")
        self.model.set_weights(self.best_weights)
