import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class FeatureBuilderBase(ABC):
    """
    Abstract base class defining the contract for all feature builder transformers.
    Implementations are responsible for deriving new columns from existing DataFrame columns.
    All feature builders expose their required and produced column names for pipeline validation.
    """

    ###############################################################
    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the feature and add it as a new column to the DataFrame.

        :param df: Input DataFrame containing required columns.
        :type df: pd.DataFrame
        :returns: DataFrame with new feature columns added.
        :rtype: pd.DataFrame
        :raises ValueError: If required columns are missing or fail type validation.
        """
        pass

    ###############################################################
    @abstractmethod
    def get_feature_names_in(self) -> list[str]:
        """
        Return the input column names this builder requires.

        :returns: List of required input column name strings.
        :rtype: list[str]
        """
        pass

    ###############################################################
    @abstractmethod
    def get_feature_names_out(self) -> list[str]:
        """
        Return the output column names this builder produces.

        :returns: List of produced output column name strings.
        :rtype: list[str]
        """
        pass