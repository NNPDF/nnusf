# -*- coding: utf-8 -*-
import json
import logging
from dataclasses import dataclass

import tensorflow as tf

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


@dataclass
class TrainingInfo:
    """Class for storing info to be shared among callbacks (in particular prevents evaluating multiple times for each individual callback)"""

    vl_chi2: float
    chix: list
    vl_chi2_history: dict


class GetTrainingInfo(tf.keras.callbacks.Callback):
    """Fill the TrainingInfo class. This is the first callback being called at each epoch"""

    def __init__(self, vl_model, kinematics, vl_expdata, traininfo_class):
        self.vl_chi2 = None
        self.vl_model = vl_model
        self.kinematics = kinematics
        self.vl_expdata = vl_expdata
        self.traininfo_class = traininfo_class
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        chix = self.vl_model.evaluate(
            self.kinematics, y=self.vl_expdata, verbose=0
        )
        vl_chi2 = chix[0] if isinstance(chix, list) else chix
        self.traininfo_class.vl_chi2 = vl_chi2
        self.traininfo_class.chix = chix


class AdaptLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, tr_dapts):
        super().__init__()
        self.loss_value = 1e5
        self.nbdpts = sum(tr_dapts.values())

    def on_batch_end(self, batch, logs={}):
        """Update value of LR after each epochs"""
        self.loss_value = logs.get("loss") / self.nbdpts

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer does not have LR attribute.")
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        scheduled_lr = modify_lr(self.loss_value, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        vl_model,
        tr_dpts,
        vl_dpts,
        patience_epochs,
        chi2_threshold,
        table,
        live,
        print_rate,
        traininfo_class,
    ):
        super().__init__()
        self.vl_model = vl_model
        self.live = live
        self.table = table
        self.best_epoch = None
        self.best_chi2 = None
        self.best_weights = None
        self.threshold = chi2_threshold
        self.tr_dpts = tr_dpts
        self.vl_dpts = vl_dpts
        self.patience_epochs = patience_epochs
        self.tot_vl = sum(vl_dpts.values())
        self.print_rate = print_rate
        self.traininfo_class = traininfo_class

    def on_epoch_end(self, epoch, logs={}):
        chi2 = self.traininfo_class.vl_chi2
        chix = self.traininfo_class.chix
        if self.best_chi2 == None or chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()

        if (epoch % self.print_rate) == 0:
            lr = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
            self.table = chi2_logs(
                logs, chix, self.tr_dpts, self.vl_dpts, epoch, lr
            )
            self.live.update(self.table, refresh=True)

        epochs_since_best_vl_chi2 = epoch - self.best_epoch
        check_val = epochs_since_best_vl_chi2 > self.patience_epochs
        if check_val and ((self.best_chi2 / self.tot_vl) <= self.threshold):
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        _logger.info(f"best epoch: {self.best_epoch}")
        self.model.set_weights(self.best_weights)


class LogTrainingInfo(tf.keras.callbacks.Callback):
    def __init__(self, replica_dir, traininfo_class):
        self.chi2_history_file = replica_dir / "chi2_history.json"
        self.traininfo_class = traininfo_class
        self.traininfo_class.vl_chi2_history = {}
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.traininfo_class.vl_chi2_history[
            epoch
        ] = self.traininfo_class.vl_chi2

    def on_train_end(self, logs=None):
        with open(
            f"{self.chi2_history_file}", "w", encoding="UTF-8"
        ) as ostream:
            json.dump(
                self.traininfo_class.vl_chi2_history,
                ostream,
                sort_keys=True,
                indent=4,
            )
