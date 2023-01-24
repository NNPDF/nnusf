# -*- coding: utf-8 -*-
import json
import logging

import tensorflow as tf

from .utils import chi2_logs

_logger = logging.getLogger(__name__)

CHI2_HISTORY_FILE = "chi2_history.yaml"


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
        self.traininfo_class.tr_chi2 = logs.get("loss")
        self.traininfo_class.loss_value = (
            self.traininfo_class.tr_chi2 / self.traininfo_class.nbdpts
        )


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
            self.table = chi2_logs(
                logs,
                self.traininfo_class.chix,
                self.traininfo_class.tr_dpts,
                self.traininfo_class.vl_dpts,
                epoch,
            )
            self.live.update(self.table, refresh=True)


class LogTrainingHistory(tf.keras.callbacks.Callback):
    def __init__(self, replica_dir, traininfo_class, log_freq):
        self.replica_dir = replica_dir
        self.traininfo_class = traininfo_class
        self.log_freq = log_freq
        self.traininfo_class.chi2_history = {}
        if (self.replica_dir / CHI2_HISTORY_FILE).exists():
            (self.replica_dir / CHI2_HISTORY_FILE).unlink()
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.traininfo_class.chi2_history[epoch] = {
            "vl": self.traininfo_class.vl_loss_value,
            "tr": self.traininfo_class.loss_value,
        }
        if epoch % self.log_freq == 0:
            with open(
                self.replica_dir / CHI2_HISTORY_FILE, "a", encoding="UTF-8"
            ) as f:
                f.write(f"{epoch}:\n")
                f.write(f"  tr: {self.traininfo_class.loss_value}\n")
                f.write(f"  vl: {self.traininfo_class.vl_loss_value}\n")

    def on_train_end(self, logs={}):
        # Save number of datapoints for the reports
        datapoints_per_dataset = {
            k: v + self.traininfo_class.vl_dpts[k]
            for k, v in self.traininfo_class.tr_dpts.items()
            if k in self.traininfo_class.vl_dpts
        }
        # Save the chi2/Ndat for the individual dataset
        chi2s_per_dataset = {
            k.strip("_loss"): v / self.traininfo_class.tr_dpts[k.strip("_loss")]
            for k, v in logs.items()
            if k.strip("_loss") in self.traininfo_class.tr_dpts
        }

        # write info of best model to log
        final_results = {
            "chi2s_per_dataset": chi2s_per_dataset,
            "dtpts_per_dataset": datapoints_per_dataset,
            "best_tr_chi2": self.traininfo_class.loss_value,
            "best_vl_chi2": self.traininfo_class.best_chi2
            / self.traininfo_class.tot_vl,
            "best_epochs": self.traininfo_class.best_epoch,
        }

        with open(
            f"{self.replica_dir}/fitinfo.json", "w", encoding="UTF-8"
        ) as ostream:
            json.dump(final_results, ostream, sort_keys=True, indent=4)
