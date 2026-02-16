from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Tuple

from models.models import BuildParams, TrainingParams


class ModelBuilderBase(ABC):
    """
    Base abstraction for model builders responsible for constructing
    and training a model from a prepared training DataFrame.
    """

    @abstractmethod
    def build_and_train_model(
        self,
        train_df: pd.DataFrame,
        featureCols: list[str],
        buildConfig: BuildParams,
        trainConfig: TrainingParams,
        target_col: str
    ) -> Tuple[Any, Any]:
        """
        Build and train a model.

        :param train_df: Fully prepared training DataFrame.
        :param featureCols: List of feature column names.
        :param buildConfig: Model architecture parameters.
        :param trainConfig: Training hyperparameters.
        :param target_col: Target column name.
        :returns: Tuple of (trained model, training history).
        """
        raise NotImplementedError
    #--------------------------#
