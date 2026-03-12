import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class SampleBuilderBase(ABC):
    """
    Abstract base class defining the contract for negative sample builders.
    Implementations are responsible for expanding a training DataFrame
    with synthetic negative samples to balance the dataset.
    """

    ###############################################################
    @abstractmethod
    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build and insert negative samples into the DataFrame.

        :param df: Input DataFrame containing positive purchase samples.
        :type df: pd.DataFrame
        :returns: Expanded DataFrame with negative samples inserted.
        :rtype: pd.DataFrame
        """
        pass