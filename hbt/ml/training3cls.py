# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

from typing import Any

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column
from training_pi_3cls import input_names, data_path, particle_names, open_parquet_files, combine_particle_columns, combine_weights_columns, remove_empty_floats

ak = maybe_import("awkward")

law.contrib.load("tensorflow")

hyperparameters = {
    "folds": 5,
    "epochs": 500,
}

# input and target features
configuration_dict = {
    "input_features": (
        "pion_neg",
        "pion_pos",
    ),
    "target_features": (
        "classes",
    ),
}

# combine configuration dictionary
configuration_dict.update(hyperparameters)

# init model instance with config dictionary
test_model = TestModel.derive(cls_name="test_model", cls_dict=configuration_dict)

class TestModel(MLModel):
    # shared between all model instances
    datasets: dict = {
        "datasets_name": [
            "hh_ggf_bbtautau",
            "tt_dl_powheg",
            "dy_lep_pt50To100_amcatnlo",

        ],
    }

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)


# class TestModel(MLModel):

#     def setup(self):
#         # dynamically add variables for the quantities produced by this model
#         if f"{self.cls_name}"not in self.config_inst.variables:
#             self.config_inst.add_variable(
#                 name=f"{self.cls_name}.n_muon",
#                 null_value=-1,
#                 binning=(4, -1.5, 2.5),
#                 x_title="Predicted number of muons",
#             )
#             self.config_inst.add_variable(
#                 name=f"{self.cls_name}.n_electron",
#                 null_value=-1,
#                 binning=(4, -1.5, 2.5),
#                 x_title="Predicted number of electrons",
#             )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {
            config_inst.get_dataset("hh_ggf_bbtautau"),
            config_inst.get_dataset("tt_dl_powheg"),
            config_inst.get_dataset("dy_lep_pt50To100_amcatnlo"),
        }
    from IPython import embed; embed()
    inputs = [f"{particle}.{input_name}" for particle in particle_names for input_name in input_names]
    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {
            set(inputs) #can I leave it like this?
        }

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {
            f"{self.cls_name}.pion_neg", f"{self.cls_name}.pion_pos",
        }

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"training_3_cls_fold_{task.fold}of{self.folds}", dir=True)

    def open_model(self, target: law.FileSystemDirectoryTarget):
        loaded_model = tf.keras.models.load_model(target.path)
        return loaded_model

    def train(
        self,
        task: law.Task,
        input: dict[str, list[law.FileSystemFileTarget]],
        output: law.FileSystemDirectoryTarget,
    ) -> None:
        tf = maybe_import("tensorflow")

        kernel_regularizer = tf.keras.regularizers.L2(0.01)
        # define NN
        model = keras.Sequential(
            [
                layers.InputLayer(input_shape=(8,)),
                layers.Dense(128, name="layer1", kernel_regularizer=kernel_regularizer),
                layers.BatchNormalization(),
                layers.Activation(tf.nn.elu),
                layers.Dense(128, name="layer2", kernel_regularizer=kernel_regularizer),
                layers.BatchNormalization(),
                layers.Activation(tf.nn.elu),
                layers.Dense(128, name="layer3", kernel_regularizer=kernel_regularizer),
                layers.BatchNormalization(),
                layers.Activation(tf.nn.elu),
                layers.Dense(3, activation="softmax", name="output"),
            ]
        )

        # TODO: get inputs and weights, create targets and combine everything in one tensor
        particle_names = ["pion_neg",
                "pion_pos"]
 
        features = [
                'eta', 
                # 'mass', 
                'phi', 
                'pt', 
                'pdgId', 
                # 'status', 
                # 'statusFlags',
                'zfrac',
                
            ]
        def combine_particle_columns(ak_array, input_names,features):
            return ak.concatenate([ak_array[name][features] for name in input_names],axis=0)

        def combine_weights_columns(ak_array, weight):
            return ak.concatenate([ak_array[weight],ak_array[weight]],axis=0) 
        if __name__ == '__main__':
            input_array_combined = ak.Array([])
            target_array_combined = ak.Array([])
            sample_weights = ak.Array([])
    
            for dataset in data_path.keys():
                input_array, target_array, weights_array = prepare_input_data(dataset)
                input_array_combined = ak.concatenate([input_array_combined, input_array], axis=0)
                target_array_combined = ak.concatenate([target_array_combined, target_array], axis=0)
                weight = (weights_array/ak.sum(weights_array))*1E5
                # bc of manuell weights
                sample_weights = ak.concatenate([sample_weights, weight], axis=0)
        
            # set manuell weights
            labels = ak.argmax(target_array_combined, axis=1)
            unique_labels = set(labels)
            new_weights = np.zeros(len(labels), dtype=np.float32)
    
            for label in unique_labels:
                mask = labels==label
                new_weights[mask] = len(labels) / (len(unique_labels) * ak.sum(mask)) 
   
    
            tf_dataset_combined = change_to_tf_datasets(input_array_combined, target_array_combined, new_weights)
            train, test = split_dataset(tf_dataset_combined)       

        # TODO: setup training with model.compile
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=False)

        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # lr_metric = get_lr_metric(optimizer)
        model.compile(optimizer=optimizer, 
                
                    loss=loss_fn,
                
                    metrics=[
                        'accuracy',
                        'categorical_accuracy',
                        'categorical_crossentropy',
                        # lr_metric,
                    ],
                    weighted_metrics=[
                        'accuracy',
                        'categorical_accuracy',
                        'categorical_crossentropy',
                    ])

        # TODO: DO training with model.fit
        history = model.fit(train, validation_data=test, epochs=500, callbacks=[early_stopping])

        # the output is just a single directory target
        output.dump(model, formatter="tf_keras_model")

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = False,
    ) -> ak.Array:
        # fake evaluation
        events = set_ak_column(events, f"{self.cls_name}.n_muon", 1)
        events = set_ak_column(events, f"{self.cls_name}.n_electron", 1)

        return events


# usable derivations
test_model = TestModel.derive("test_model", cls_dict={"folds": 3})
