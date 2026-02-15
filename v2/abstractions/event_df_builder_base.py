import pandas as pd
from abc import ABC, abstractmethod


###############################################################
class EventDfBuilderBase(ABC):
    """
    Abstract base class defining the contract for event DataFrame builders.
    Implementations are responsible for constructing a DataFrame representing
    a specific event context such as a prediction date, a vendor receipt, or a trip.
    Each implementation defines its own required inputs via constructor injection.
    """

    ###############################################################
    @abstractmethod
    def build_df(self, *args, **kwargs) -> pd.DataFrame:
        """
        Build an event DataFrame.
        Each implementation defines its own required arguments via constructor injection
        or method parameters appropriate to its event context.

        :returns: DataFrame representing the event context.
        :rtype: pd.DataFrame
        """
        pass