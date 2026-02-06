import pandas as pd
import tensorflow as tf
from typing import Dict, Any


class ModelArtifacts:

    def __init__(
        self,
        model: tf.keras.Model,
        training_df: pd.DataFrame,
        normalization_params: Dict[str, Any],
        history: Dict[str, Any],
        build_params: Dict[str, Any],
        train_params: Dict[str, Any],
    ):
        self.model = model
        self.training_df = training_df
        self.normalization_params = normalization_params
        self.history = history
        self.build_params = build_params
        self.train_params = train_params
