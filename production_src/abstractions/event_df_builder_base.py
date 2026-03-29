import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class EventDfBuilderBase(ABC):
    """
    Abstract base class defining the contract for event DataFrame builders.

    Implementations are responsible for constructing a DataFrame representing
    a specific event context (e.g. vendor receipts, prediction dates, trips).

    Builders are stateless and receive required source paths at execution time.
    """

    ###############################################################
    @abstractmethod
    def build_df(self, sourcePaths: dict) -> pd.DataFrame:
        """
        Build an event DataFrame using the provided source paths.

        :param sourcePaths: Dictionary of named source paths relevant to the builder.
        :type sourcePaths: dict

        :returns: DataFrame representing the event context.
        :rtype: pd.DataFrame
        """
        pass
