# -*- coding: utf-8 -*-
import tensorflow as tf


class SFModel(tf.keras.Model):
    def __init__(self, feature_scaling_functions=None, **kwargs):
        self.feature_scaling_functions = feature_scaling_functions

        def scaling_func(inputs):
            scaled = [
                fs_func(inputs[:, :, i])
                for fs_func, i in zip(self.feature_scaling_functions, range(3))
            ]
            return tf.stack(scaled, axis=-1)

        self.scaling_func = scaling_func
        super().__init__(**kwargs)

    def fit(self, kinematics_array, **kwargs):
        scaled_kinematics_array = [
            self.scaling_func(dataset_arr) for dataset_arr in kinematics_array
        ]
        super().fit(scaled_kinematics_array, **kwargs)

    def predict(self, kinematics_array, **kwargs):
        scaled_kinematics_array = [
            self.scaling_func(dataset_arr) for dataset_arr in kinematics_array
        ]
        super().predict(scaled_kinematics_array, **kwargs)
