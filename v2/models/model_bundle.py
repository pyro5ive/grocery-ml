import pandas as pd
import tensorflow as tf
from models.models import *

class ModelBundle:

    def __init__(
        self,
        model: tf.keras.Model,
        training_df: pd.DataFrame,
        itemMappingDf: pd.DataFrame,
        normalization_params: Dict[str, Any],
        history: Dict[str, Any],
        build_params: BuildParams,
        train_params: TrainingParams
    ):
        self.model = model
        self.training_df = training_df
        self.normalization_params = normalization_params
        self.history = history
        self.build_params = build_params
        self.train_params = train_params
        self.itemMappingDf = itemMappingDf
