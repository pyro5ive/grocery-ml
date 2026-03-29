import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class DfFilterBase(ABC):
    """
    Abstract base class defining the contract for DataFrame filters.
    Implementations are responsible for cleaning, consolidating, or
    reshaping a DataFrame prior to feature pipeline execution.
    """

    ###############################################################
    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the filter operation to the DataFrame.

        :param df: Input DataFrame to filter.
        :type df: pd.DataFrame
        :returns: Filtered DataFrame.
        :rtype: pd.DataFrame
        """
        pass