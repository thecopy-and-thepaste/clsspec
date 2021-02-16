import argparse
import os
import json
import shutil

import pandas as pd
import numpy as np
import tensorflow as tf

from pprint import pprint
from pathlib import Path

# Remember to update the PYTHON_PATH to
# export PYTHONPATH=`pwd`:`pwd`/conabio_ml_text/conabio_ml:`pwd`/conabio_ml_text
from conabio_ml.pipeline import Pipeline

from conabio_ml_text.datasets.dataset import Dataset, Partitions
from conabio_ml_text.preprocessing.preprocessing import Tokens, PreProcessing
from conabio_ml_text.preprocessing.transform import Transform

from conabio_ml_text.trainers.bcknds.tfkeras import TFKerasTrainer, TFKerasTrainerConfig
from conabio_ml_text.trainers.bcknds.tfkeras import CHECKPOINT_CALLBACK, TENSORBOARD_CALLBACK

from conabio_ml_text.utils.constraints import TransformRepresentations as TR, LearningRates as LR
from conabio_ml_text.trainers.builders import create_learning_rate

from conabio_ml.evaluator.generic.evaluator import Evaluator, Metrics

from conabio_ml.utils.logger import get_logger, debugger

from utils import nchars, datagen
from model import LSTMModel

log = get_logger(__name__)
debug = debugger.debug


class Helper:
    def __init__(self, config) -> None:
        self.config = config

    def report_vocab(self, dataset: Dataset.DatasetType, **kwargs) -> Dataset.DatasetType:
        self.config["layers"]["embedding"]["V"] = len(
            dataset.representations[TR.VOCAB])

        print("Reporting vocab")
        print("_________________")
        print(self.config["layers"]["embedding"])
        return dataset


def run(config_file: str):
    Tokens.UNK_TOKEN = Tokens.UNK_TOKEN * 3

    config = {}
    with open(config_file) as _f:
        config = json.load(_f)

    dataset_filepath = Path(f"{config['dataset']}")
    results_path = Path(f"pipelines/results")

    # Model layers
    layers = config["layers"]

    # Since the process of split/proprocess is done in the pipeline, we need a helper
    # to report variables not computed early. Like, vocab_size
    vocab_size = layers["embedding"]["V"]
    helper = Helper(config=config)
    # Params

    # Learning rate
    initial_lr = config["params"]["initial_learning_rate"]
    # Decay steps for the lr
    decay_steps = config["params"]["decay_steps"]
    # Batch size
    batch_size = config["params"]["batch_size"]
    # Epochs of training
    epochs = config["params"]["epochs"]

    TRAIN_REPRESENTATION = TR.DATA_GENERATORS
    lr_schedule = create_learning_rate({"initial_learning_rate": initial_lr,
                                        "decay_steps": decay_steps},
                                       learning_rate_name=LR.EXPONENTIAL_LR)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    pipeline = Pipeline(results_path,
                        name=f"clsspec")\
        .add_process(name="trainer_config",
                     action=TFKerasTrainerConfig.create,
                     args={
                         "config": {
                             "strategy": None,
                             "callbacks": {
                                 CHECKPOINT_CALLBACK: {
                                     "filepath": os.path.join(results_path, "checkpoints"),
                                     "save_best_only": True
                                 },
                                 TENSORBOARD_CALLBACK: {
                                     "log_dir": os.path.join(results_path, "tb_logs")
                                 }
                             }
                         }
                     })\
        .add_process(name="dataset",
                     action=Dataset.from_csv,
                     reportable=True,
                     args={
                         "source_path": dataset_filepath
                     })\
        .add_process(name="split_dataset",
                     action=Dataset.split,
                     inputs_from_processes=["dataset"],
                     args={
                         "train_perc": 0.8,
                         "test_perc": 0.1,
                         "val_perc": 0.1
                     })\
        .add_process(name="preprocess_dataset",
                     action=PreProcessing.preprocess,
                     inputs_from_processes=["split_dataset"],
                     args={
                         "build_vocab": True,
                         "preprocess_fn": nchars,
                         "preprocess_args": {
                             "fields": ["item"],
                             "func_args": {
                                 "pad_size": -1,
                                 "nchar_size": 3,
                                 "unk_token": Tokens.UNK_TOKEN
                             }
                         }
                     })\
        .add_process(name="transform_to_datagen",
                     action=Transform.as_data_generator,
                     inputs_from_processes=["preprocess_dataset"],
                     args={
                         "vocab": None,
                         "shuffle": True,
                         "categorical_labels": True,
                         "transform_args": {
                             "pad_length": layers["input"]["T"],
                             "unk_token": Tokens.UNK_TOKEN,
                             "batch_size": batch_size
                         },
                          "data_generator":datagen
                     })\
        .add_process(name="report_vocab",
                     action=helper.report_vocab,
                     inputs_from_processes=["transform_to_datagen"],
                     args={

                     })\
        .add_process(name="create_classifier",
                     action=LSTMModel.create,
                     args={
                         "model_config": {
                             "CLASSIFIER": {
                                 "layers": {
                                     "input": layers["input"],
                                     "embedding": layers["embedding"],
                                     "lstm": layers["lstm"],
                                     "dense": layers["dense"]
                                 }
                             }}
                     })\
        .add_process(name="train_classifier",
                     action=TFKerasTrainer.train,
                     reportable=True,
                     inputs_from_processes=["report_vocab",
                                            "create_classifier",
                                            "trainer_config"],
                     args={
                         "train_config": {
                             "CLASSIFIER": {
                                 "representation": TRAIN_REPRESENTATION,
                                 'optimizer': tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                                 'loss': tf.keras.losses.CategoricalCrossentropy(),
                                 "batch_size": None,
                                 "epochs": epochs,
                                 "metrics": ["accuracy"]
                             }}
                     })\
        .add_process(name="predict_classifier",
                     action=LSTMModel.predict,
                     inputs_from_processes=["train_classifier",
                                            "report_vocab"],
                     reportable=True,
                     args={
                         "execution_config": None,
                         "prediction_config": {
                             #  "pred_converter_fn": multilabel_converter(multilabel_th),
                             "sparse_predictions": False,
                             "ommit_uniques": True
                         }
                     })\
        .add_process(name="evaluate",
                     action=Evaluator.eval,
                     inputs_from_processes=["transform_to_datagen",
                                            "predict_classifier"],
                     reportable=True,
                     args={
                         "eval_config": {
                             "dataset_partition": Partitions.TEST,
                             "metrics_set": {
                                 Metrics.Sets.MULTICLASS: {
                                     'per_class': True,
                                     'average': 'macro',
                                     "zero_division": 1.0
                                 }
                             }}
                     })

    pipeline.run(report_pipeline=False)
    shutil.copy(config_file, Path(pipeline.path) / "config.json")
    shutil.copy("code/pipeline.py", Path(pipeline.path) / "pipeline.py")
    shutil.copy("code/model.py", Path(pipeline.path) / "model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-f", "--config_file",
    #                     help="Config file for the queries. See TODO:")
    parser.add_argument("-d", "--debug",  action='store_true',
                        help="Enables debug mode to trace the progress of your searching")
    parser.add_argument("-c", "--config",
                        help="Config file")

    ARGS = parser.parse_args()
    debugger.create(ARGS.debug, True)

    run(ARGS.config)
