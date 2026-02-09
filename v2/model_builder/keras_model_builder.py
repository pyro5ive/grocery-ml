import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split

from models.models import *

class KerasModelBuilder:

    featureColCount: int
    featureCols: list[str]
    buildConfig: BuildParams
    trainConfig: TrainingParams
    trainDf: pd.DataFrame

    def __init__(this):
        this.logger = logging.getLogger(__name__)

    #==========================================================================================#
    def build_and_train_model(
        this,
        train_df: pd.DataFrame,
        featureCols: list[str],
        buildConfig: BuildParams,
        trainConfig: TrainingParams,
        target_col: str
    ):

        this.featureColCount = len(featureCols)
        this.featureCols = featureCols
        this.trainConfig = trainConfig
        this.buildConfig = buildConfig
        this.trainDf = train_df

        model = this._build_model()
        history = this._train_model(model, target_col)

        return model, history

    #==========================================================================================#
    def _build_model(this):

        item_count = int(this.trainDf["itemId"].max()) + 1

        num_in = layers.Input(shape=(this.featureColCount,))
        item_in = layers.Input(shape=(), dtype="int32")

        emb = layers.Embedding(
            input_dim=item_count,
            output_dim=this.buildConfig.embeddingDimCount,
            name="item_embedding"
        )(item_in)

        x = layers.Concatenate()([num_in, layers.Flatten()(emb)])

        for spec in this.buildConfig.layers:
            x = layers.Dense(spec.units, activation=spec.activation)(x)

        outputLayer = layers.Dense(
            1,
            activation=this.buildConfig.outputActivation
        )(x)

        model = models.Model(inputs=[num_in, item_in], outputs=outputLayer)

        if this.buildConfig.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=this.buildConfig.learningRate
            )
        elif this.buildConfig.optimizer == "adamw":
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=this.buildConfig.learningRate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {this.buildConfig.optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=this.buildConfig.loss,
            metrics=this.buildConfig.metrics
        )

        return model

    #==========================================================================================#
    def _train_model(this, model, target_col):

        this.logger.info("train_model()")

        df = this.trainDf

        x_feat = df[this.featureCols].to_numpy(np.float32)
        x_item = df["itemId"].to_numpy(np.int32)
        y = df[target_col].to_numpy(np.float32)

        x_feat_tr, x_feat_te, x_item_tr, x_item_te, y_tr, y_te = train_test_split(
            x_feat, x_item, y, test_size=0.2, random_state=42
        )

        history = model.fit(
            [x_feat_tr, x_item_tr],
            y_tr,
            validation_data=([x_feat_te, x_item_te], y_te),
            epochs=this.trainConfig.epochs,
            batch_size=this.trainConfig.batchSize,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        return history.history
