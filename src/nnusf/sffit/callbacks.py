# -*- coding: utf-8 -*-
import json
import logging
from dataclasses import dataclass
from typing import Union

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
class TrainingStatusInfo:
    """Class for storing info to be shared among callbacks
    (in particular prevents evaluating multiple times for each individual
    callback).
    """

    tr_dpts: int
    vl_dpts: int
    best_chi2: Union[float, None] = None
    vl_chi2: Union[float, None] = None
    chix: Union[list, None] = None
    chi2_history: Union[dict, None] = None
    loss_value: float = 1e5
    vl_loss_value: Union[float, None] = None
    best_epoch: Union[int, None] = None

    def __post_init__(self):
        self.tot_vl = sum(self.vl_dpts.values())
        self.nbdpts = sum(self.tr_dpts.values())


class GetTrainingInfo(tf.keras.callbacks.Callback):
    """Fill the TrainingInfo class.
    This is the first callback being called at each epoch
    """

    def __init__(self, vl_model, kinematics, vl_expdata, traininfo_class):
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
        self.traininfo_class.vl_loss_value = (
            vl_chi2 / self.traininfo_class.tot_vl
        )


class AdaptLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, train_info_class):
        super().__init__()
        self.train_info_class = train_info_class

    def on_batch_end(self, batch, logs={}):
        """Update value of LR after each epochs"""
        self.train_info_class.tr_chi2 = logs.get("loss")
        self.train_info_class.loss_value = (
            self.train_info_class.tr_chi2 / self.train_info_class.nbdpts
        )

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer does not have LR attribute.")
        lr = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        scheduled_lr = modify_lr(self.train_info_class.loss_value, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        vl_model,
        patience_epochs,
        chi2_threshold,
        traininfo_class,
    ):
        super().__init__()
        self.vl_model = vl_model
        self.best_weights = None
        self.threshold = chi2_threshold
        self.patience_epochs = patience_epochs
        self.traininfo_class = traininfo_class

    def on_epoch_end(self, epoch, logs=None):
        chi2 = self.traininfo_class.vl_chi2
        if (
            self.traininfo_class.best_chi2 == None
            or chi2 < self.traininfo_class.best_chi2
        ):
            self.traininfo_class.best_chi2 = chi2
            self.traininfo_class.best_epoch = epoch
            self.best_weights = self.model.get_weights()

        epochs_since_best_vl_chi2 = epoch - self.traininfo_class.best_epoch
        check_val = epochs_since_best_vl_chi2 > self.patience_epochs
        if check_val and (
            (self.traininfo_class.best_chi2 / self.traininfo_class.tot_vl)
            <= self.threshold
        ):
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        _logger.info(f"best epoch: {self.traininfo_class.best_epoch}")
        self.model.set_weights(self.best_weights)


class LiveUpdater(tf.keras.callbacks.Callback):
    def __init__(self, print_rate, traininfo_class, table, live):
        self.print_rate = print_rate
        self.traininfo_class = traininfo_class
        self.table = table
        self.live = live
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.print_rate) == 0:
            lr = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
            self.table = chi2_logs(
                logs,
                self.traininfo_class.chix,
                self.traininfo_class.tr_dpts,
                self.traininfo_class.vl_dpts,
                epoch,
                lr,
            )
            self.live.update(self.table, refresh=True)


class LogTrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, replica_dir, traininfo_class):
        self.replica_dir = replica_dir
        self.traininfo_class = traininfo_class
        self.traininfo_class.chi2_history = {}
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.traininfo_class.chi2_history[epoch] = {
            "vl": self.traininfo_class.vl_loss_value,
            "tr": self.traininfo_class.loss_value,
        }

    def on_train_end(self, logs=None):
        # write log of chi2 history
        with open(
            f"{self.replica_dir}/chi2_history.json", "w", encoding="UTF-8"
        ) as ostream:
            json.dump(
                self.traininfo_class.chi2_history,
                ostream,
                sort_keys=True,
                indent=4,
            )

        # write info of best model to log
        final_results = {
            "best_tr_chi2": self.traininfo_class.loss_value,
            "best_vl_chi2": self.traininfo_class.best_chi2
            / self.traininfo_class.tot_vl,
            "best_epochs": self.traininfo_class.best_epoch,
        }
        with open(
            f"{self.replica_dir}/fitinfo.json", "w", encoding="UTF-8"
        ) as ostream:
            json.dump(final_results, ostream, sort_keys=True, indent=4)
