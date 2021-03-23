"""

This script defines all the model to use in the pipeline/clsspec. 
Note that is based in the `layer_config` config json.

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout

from conabio_ml_text.trainers.bcknds.tfkeras_models import TFKerasRawDataModel
from conabio_ml_text.trainers.bcknds.tfkeras import TFKerasBaseModel

from conabio_ml.utils.logger import get_logger

log = get_logger(__name__)


class LSTMModel(TFKerasRawDataModel):

    @classmethod
    def create_model(cls,
                     layer_config: dict) -> TFKerasBaseModel.TFKerasModelType:
        try:
            layers = layer_config["layers"]

            input_layer = layers["input"]
            embedding = layers["embedding"]
            lstm_1 = layers["lstm"]
            dense = layers["dense"]

            i = Input(shape=(input_layer["T"], ))
            x = Embedding(input_dim=embedding["V"],
                          output_dim=embedding["D"])(i)
            x = LSTM(units=lstm_1["M"],
                     recurrent_dropout=lstm_1["recurrent_dropout"])(x)
            x = Dropout(lstm_1["dropout"])(x)

            x = Dense(units=dense["K"],
                      activation="softmax")(x)  # Because we convert samples to one-hot

            model = Model(i, x)

            return model
        except Exception as ex:
            log.exception(ex)
            raise
