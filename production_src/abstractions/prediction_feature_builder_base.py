from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime


class PredictionFeatureBuilderBase(ABC):
    """
    Base for prediction-time feature builders.
    These are not part of the training feature pipeline.
    """

    @abstractmethod
    def build_df(
        self,
        df: pd.DataFrame,
        prediction_date: datetime
    ) -> pd.DataFrame:
        raise NotImplementedError
    #--------------------------#
