from abc import ABC, abstractmethod
import pandas as pd


class TargetColumnBuilderBase(ABC):
    """
    Base abstraction for building training target/label columns.

    Implementations are responsible for adding one or more target
    columns to the DataFrame. These builders are intended for
    training only and must not be used during prediction.
    """

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target/label column(s) to the DataFrame.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :returns: DataFrame with target column(s) added.
        :rtype: pd.DataFrame
        """
        raise NotImplementedError
    #--------------------------#
