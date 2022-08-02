# -*- coding: utf-8 -*-
import tensorflow as tf


class SFModel(tf.keras.Model):
    def __init__(self, feature_scaling_functions=None, **kwargs):
        self.feature_scaling_functions = feature_scaling_functions
        super().__init__(**kwargs)

    def _scaling_func(self, inputs):
        scaled = [
            fs_func(inputs[:, :, i])
            for fs_func, i in zip(self.feature_scaling_functions, range(3))
        ]
        return tf.stack(scaled, axis=-1)

    def _scale_kinematics(self, kinematics):
        return [self._scaling_func(dataset_arr) for dataset_arr in kinematics]

    def fit(self, kinematics_array, **kwargs):
        scaled_kinematics_array = self._scale_kinematics(kinematics_array)
        super().fit(scaled_kinematics_array, **kwargs)

    def predict(self, kinematics_array, **kwargs):
        scaled_kinematics_array = self._scale_kinematics(kinematics_array)
        super().predict(scaled_kinematics_array, **kwargs)

    # def evaluate(self, kinematics_array, **kwargs):
    #     scaled_kinematics_array = self._scale_kinematics(kinematics_array)
    #     super().evaluate(scaled_kinematics_array, **kwargs)
